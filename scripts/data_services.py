import glob
from fractions import Fraction
from typing import List, Tuple

import numpy as np
from music21 import converter, instrument, note, chord, stream

import constants as const
from utils import *


def translate_midis(data_path: str, save_dir: str, with_timing=True) -> None:
    """ Converts raw MIDIs into the list of parts of notes.
    Parts consist of notes only (there is no direct representation for chords,
    however, this is possible with with_timing = True).
    A note is stored in a part as a string with the format:
    (relative_offset + FIELD_SEPARATOR + duration + FIELD_SEPARATOR + note_pitch).
    The list of parts along with dataset vocabulary and params are serialized to files.
    :param data_path: path to raw MIDI data, blobs are allowed
    :param save_dir: root directory for serialized dataset, vocabs and params
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
                        string_note += str(relative_offset) + const.FIELD_SEPARATOR
                        string_note += str(item.duration.quarterLength) + const.FIELD_SEPARATOR
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

    all_notes_flat = []
    for part in all_notes:
        all_notes_flat.extend(part)

    vocab = sorted(set(all_notes_flat))
    note2idx = {str_note: idx for idx, str_note in enumerate(vocab)}
    idx2note = np.array(vocab)

    # Debug ###############
    sample_notes_as_ints = np.array([note2idx[n] for n in all_notes_flat[:15]])
    log('d', 'Notes as ints: ' + str(sample_notes_as_ints))
    log('d', 'Ints as notes [mapping]: ' + str(idx2note[sample_notes_as_ints]))
    log('d', 'Ints as notes [original]: ' + str(np.array(all_notes_flat[:15])))
    # Debug end ###########

    save(all_notes, os.path.join(save_dir, 'translated_midis.pickle'))
    save(note2idx, os.path.join(save_dir, 'note2idx.pickle'))
    save(idx2note, os.path.join(save_dir, 'idx2note.pickle'))
    save({'WITH_TIMING': with_timing}, os.path.join(save_dir, 'params_py_dict.pickle'))


def load_translated_dataset(dataset_path: str) -> Tuple[List[List[str]], dict, np.ndarray, dict]:
    all_notes = load(os.path.join(dataset_path, 'translated_midis.pickle'))
    note2idx = load(os.path.join(dataset_path, 'note2idx.pickle'))
    idx2note = load(os.path.join(dataset_path, 'idx2note.pickle'))
    dataset_params = load(os.path.join(dataset_path, 'params_py_dict.pickle'))
    return all_notes, note2idx, idx2note, dataset_params


def create_dataset(load_translated_dataset_result: Tuple[List[List[str]], dict, np.ndarray, dict], seq_len: int,
                   flat=False):
    data_matrix, note2idx, idx2note, dataset_params = load_translated_dataset_result
    all_datasets = []
    # TODO tweaks for data continuity between batches (for model's stateful=True)
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
        return flat_dataset
    else:
        log('i', 'Dataset created!')
        log('i', 'Flat?: NO')
        log('i', 'Type (partial dataset): %s' % all_datasets[0])
        log('i', 'No. of batches: %s' % [len(ds) for ds in all_datasets])
        log('i', 'Vocabulary size: %d' % len(idx2note))
        return all_datasets


def save_notes_to_midi(notes: List[str], path: str) -> None:
    output_notes = []
    offset = 0
    if const.FIELD_SEPARATOR in notes[0]:  # item: relative_offset$quarter_note_duration$pitch_with_octave
        for item in notes:
            item = item.split(const.FIELD_SEPARATOR)
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


def main():
    DATA_FILES_GLOB = '01[ac]*.mid'  # Load
    TRANSLATED_DATASET_NAME = 'bach_sample'  # Save

    translate_midis(data_path=os.path.join(const.PATH_TO_RAW_MIDIS, DATA_FILES_GLOB),
                    save_dir=os.path.join(const.PATH_TO_TRANSLATED_DATASETS, TRANSLATED_DATASET_NAME),
                    with_timing=True)
    # translated_dataset = load_translated_dataset(
    #     os.path.join(const.PATH_TO_TRANSLATED_DATASETS, TRANSLATED_DATASET_NAME))
    # dataset = create_dataset(translated_dataset, seq_len=3, flat=True)


if __name__ == '__main__':
    main()
