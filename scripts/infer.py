import os

import numpy as np
import tensorflow as tf

import constants as const
import data_services
import train
import utils


def load_model(model_dir: str, vocab_size: int):
    model_params = utils.load(os.path.join(model_dir, 'model_params.pickle'))
    inference_model = train.create_model(vocab_size, model_params['EMBEDDING_DIM'], model_params['RNN_UNITS'],
                                         batch_size=1)
    inference_model.load_weights(tf.train.latest_checkpoint(model_dir))
    inference_model.build(tf.TensorShape([1, None]))
    return inference_model


def generate_music(inference_model, load_translated_dataset_result, notes_to_generate, sample_size):
    _data_matrix, _note2idx, _idx2note, _dataset_params = load_translated_dataset_result
    sample_notes = _data_matrix[0]
    random_offset = np.random.randint(len(sample_notes) - sample_size + 1)
    start_phrase = sample_notes[random_offset:random_offset + sample_size]
    model_input = [_note2idx[note] for note in start_phrase]
    model_input = tf.expand_dims(model_input, 0)

    music_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    inference_model.reset_states()
    for i in range(notes_to_generate):  # TODO rather generate with music time limit
        predictions = inference_model(model_input)
        predictions = tf.squeeze(predictions, 0)
        predictions /= temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        model_input = tf.expand_dims([predicted_id], 0)
        music_generated.append(_idx2note[predicted_id])

    return music_generated


def main():
    MODEL_NAME = 'bach_var'  # Load
    TRANSLATED_DATASET_NAME = 'bach_sample_var'  # Load
    COMPOSITION_NAME = 'test_composition_var.mid'  # Save
    SAMPLE_SIZE = 10
    NOTES_TO_GENERATE = 1000

    translated_dataset = data_services.load_translated_dataset(
        os.path.join(const.PATH_TO_TRANSLATED_DATASETS, TRANSLATED_DATASET_NAME))
    data_matrix, note2idx, idx2note, dataset_params = translated_dataset
    model = load_model(os.path.join(train.CHECKPOINTS_DIR, MODEL_NAME), idx2note.size)
    model.summary()
    generated_music = generate_music(model, translated_dataset, NOTES_TO_GENERATE, SAMPLE_SIZE)
    data_services.save_notes_to_midi(generated_music, os.path.join(const.PATH_TO_COMPOSITIONS, COMPOSITION_NAME))


if __name__ == '__main__':
    main()
