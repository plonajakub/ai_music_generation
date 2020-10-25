import glob
from fractions import Fraction
from typing import List

import numpy as np
from music21 import converter, instrument, note, chord, stream

from utils import *

FIELD_SEPARATOR = '$'


def get_notes(data_path: str, with_timing=True) -> List[List[str]]:
    """ Converts raw MIDIs into the list of parts of notes.
    Parts consist of notes only (there is no direct representation for chords,
    however, this is possible with with_timing = True).
    A note is stored in a part as a string with the format:
    (relative_offset + FIELD_SEPARATOR + duration + FIELD_SEPARATOR + note_pitch).
    :param data_path: path to raw MIDI data, blobs are allowed
    :param with_timing: when True the note string format is
    (relative_offset + FIELD_SEPARATOR + duration + FIELD_SEPARATOR + note_pitch),
    otherwise note string consists of note_pitch only
    :return: Matrix with non-empty musical parts of notes with the representation described above
    """
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


def create_dataset(data_matrix: List[List[str]], seq_len: int, flat=False):
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
        log('i', 'Dataset created!')
        log('i', 'Flat?: YES')
        log('i', 'Type: %s' % flat_dataset)
        log('i', 'No. of batches: %d' % len(flat_dataset))
        log('i', 'Vocabulary size: %d' % len(idx2note))
        print([note2idx[n] for n in data_matrix[0]][:30])  # TODO refactor debug info
        for batch in flat_dataset.take(5):
            print('Batch: %s' % str(batch))
        return flat_dataset, idx2note
    else:
        log('i', 'Dataset created!')
        log('i', 'Flat?: NO')
        log('i', 'Type (partial dataset): %s' % all_datasets[0])
        log('i', 'No. of batches: %s' % [len(ds) for ds in all_datasets])
        log('i', 'Vocabulary size: %d' % len(idx2note))
        return all_datasets, idx2note


def get_music_dataset(data_path: str, with_timing: bool, seq_len: int, flat: bool):
    notes = get_notes(data_path=data_path, with_timing=with_timing)
    return create_dataset(data_matrix=notes, seq_len=seq_len, flat=flat)


# TODO save get_notes results to a file

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
