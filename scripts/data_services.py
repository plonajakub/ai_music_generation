from typing import List
from music21 import converter, instrument, note, chord, stream
import numpy as np
import glob


def get_notes(data_path: str) -> List[List[str]]:
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
			for item in part:
				if isinstance(item, note.Note):
					if item.octave is None:
						item.octave = item.pitch.defaultOctave
					part_notes.append(item.nameWithOctave)
				elif isinstance(item, chord.Chord):
					sorted_chord = item.sortFrequencyAscending()
					soprano_pitch = sorted_chord.pitches[-1]
					if soprano_pitch.octave is None:
						soprano_pitch.octave = soprano_pitch.defaulOctave
					part_notes.append(soprano_pitch.nameWithOctave)
			if len(part_notes) != 0:
				all_notes.append(part_notes)

	return all_notes


def create_dataset(data_matrix, batch_size, seq_len):
	return np.zeros((5, 5)), ['a', 'b''c']  # TODO


def get_music_dataset(data_path: str, batch_size: int, seq_len: int):
	notes = get_notes(data_path=data_path)
	return create_dataset(data_matrix=notes, batch_size=batch_size, seq_len=seq_len)


def save_notes_to_midi(notes: List[str], path: str) -> None:
	output_notes = []
	offset = 0
	for item in notes:
		new_note = note.Note(item)
		new_note.offset = offset
		new_note.storedInstrument = instrument.Violin()
		output_notes.append(new_note)
		offset += 0.5

	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp=path)
