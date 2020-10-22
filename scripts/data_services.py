from typing import List
from music21 import converter, instrument, note, chord, stream
import numpy as np
import glob
from fractions import Fraction
import tensorflow as tf
import os
import time
from utils import *

FIELD_SEPARATOR = '$'


def get_notes(data_path: str, with_timing=True) -> List[List[str]]:
	all_notes = []
	for file in glob.glob(data_path):
		print("Parsing %s" % file)
		file_stream = converter.parse(file)

		file_parts = []
		try:  # instrument parts available
			score = instrument.partitionByInstrument(file_stream)
			for part in score.parts:
				file_parts.append(part.flat.notes)
		except AttributeError:  # no instrument parts
			file_parts.append(file_stream.flat.notes)

		for part in file_parts:
			part_notes = []

			last_offset = 0
			for item in part:
				string_note = ''
				if isinstance(item, note.Note) or isinstance(item, chord.Chord):
					if with_timing:
						relative_offset = item.offset - last_offset
						last_offset = item.offset
						string_note += str(relative_offset) + FIELD_SEPARATOR
						string_note += str(item.duration.quarterLength) + FIELD_SEPARATOR
				else:
					continue

				note_pitch = None
				if isinstance(item, note.Note):
					note_pitch = item.pitch
				else:  # isinstance(item, chord.Chord)
					sorted_chord = item.sortFrequencyAscending()
					note_pitch = sorted_chord.pitches[-1]  # Get the soprano pitch

				if note_pitch.octave is None:
					note_pitch.octave = note_pitch.implicitOctave
				string_note += note_pitch.nameWithOctave

				part_notes.append(string_note)
			if len(part_notes) != 0:
				all_notes.append(part_notes)

	return all_notes


def create_dataset(data_matrix, batch_size, seq_len, flat=False):
	all_notes = []
	for part in data_matrix:
		all_notes.extend(part)

	vocab = sorted(set(all_notes))
	note2idx = {str_note: idx for idx, str_note in enumerate(vocab)}
	idx2note = np.array(vocab)

	all_notes_as_ints = np.array([note2idx[n] for n in all_notes])
	log('d', 'Notes as ints: ' + str(all_notes_as_ints[:15]))
	log('d', 'Ints as notes [mapping]: ' + str(idx2note[all_notes_as_ints[:15]]))
	log('d', 'Ints as notes [original]: ' + str(np.array(all_notes[:15])))

	all_datasets = []
	for part in data_matrix:
		part_as_ints = [note2idx[n] for n in part]
		partial_dataset = tf.data.Dataset.from_tensor_slices(np.array(part_as_ints))
		part_sequences = partial_dataset.batch(seq_len + 1, drop_remainder=True)
		part_sequences_split = part_sequences.map(split_input_target)
		all_datasets.append(part_sequences_split)

	if flat:
		flat_dataset = all_datasets[0]
		for ds in all_datasets[1:]:
			flat_dataset = flat_dataset.concatenate(ds)
		flat_dataset = flat_dataset.batch(batch_size, drop_remainder=True)
		log('i', 'Dataset created!')
		log('i', 'Flat?: YES')
		log('i', 'Type: %s' % flat_dataset)
		log('i', 'No. of batches: %d' % len(flat_dataset))
		log('i', 'Vocabulary size: %d' % len(idx2note))
		return flat_dataset, idx2note
	else:
		for i in range(len(all_datasets)):
			all_datasets[i] = all_datasets[i].batch(batch_size, drop_remainder=True)
		log('i', 'Dataset created!')
		log('i', 'Flat?: NO')
		log('i', 'Type (partial dataset): %s' % all_datasets[0])
		log('i', 'No. of batches: %s' % [len(ds) for ds in all_datasets])
		log('i', 'Vocabulary size: %d' % len(idx2note))
		return all_datasets, idx2note


def get_music_dataset(data_path: str, batch_size: int, seq_len: int):
	notes = get_notes(data_path=data_path)
	return create_dataset(data_matrix=notes, batch_size=batch_size, seq_len=seq_len)


def save_notes_to_midi(notes: List[str], path: str) -> None:
	output_notes = []
	offset = 0
	if FIELD_SEPARATOR in notes[0]:  # item: relative_offset$quarter_note_duration$pitch_with_octave
		for item in notes:
			item = item.split(FIELD_SEPARATOR)
			new_note = note.Note(item[2])
			try:
				delta = float(item[0])
			except ValueError:
				delta = Fraction(item[0])
			offset += delta
			new_note.offset = offset
			try:
				new_note.duration.quarterLength = float(item[1])
			except ValueError:
				new_note.duration.quarterLength = Fraction(item[1])
			output_notes.append(new_note)
	else:  # item: pitch_with_octave
		for item in notes:
			new_note = note.Note(item)
			new_note.offset = offset
			output_notes.append(new_note)
			offset += 0.5

	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp=path)
