import glob
import os

import tensorflow as tf

import constants as const
import data_services
import utils
from utils import log


# TODO add more models for experiments
def create_model(vocab_size, embedding_dim, rnn_units, rnn_stateful, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=rnn_stateful),
        tf.keras.layers.Dense(vocab_size)
    ])
    model.summary()
    return model


def train_model(dataset, model, batch_size, epochs, load_model_dir, checkpoint_path):
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if len(glob.glob(os.path.join(load_model_dir, const.CHECKPOINT_NAME_GLOB))) != 0:
        model.load_weights(tf.train.latest_checkpoint(load_model_dir))
        log('i', 'Model weights have been loaded from \'' + load_model_dir + '\'')
    else:
        log('i', 'Path \'' + load_model_dir + '\' does not contain checkpoints.')
        log('i', 'Compiling new model...')
    model.compile(optimizer='adam', loss=utils.loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True)

    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


def main():
    # Data
    TRANSLATED_DATASET_NAME = 'bach_fugue_all_timing_true'  # Load
    CREATE_DATASET_FLAT = True  # TODO support sparse dataset (CREATE_DATASET_FLAT = False)
    SEQ_LEN = 500

    # Model creation
    BATCH_SIZE = 1  # Batch size = 1 would be good for stateful = True (?)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    RNN_STATEFUL = True

    # Training
    EPOCHS = 100
    MODEL_NAME = 'bach_fugue_stateful_true'  # Save
    SAVE_MODEL_DIR = os.path.join(const.PATH_TO_CHECKPOINTS, MODEL_NAME)
    LOAD_MODEL_DIR = SAVE_MODEL_DIR
    CHECKPOINT_PATH = os.path.join(SAVE_MODEL_DIR, const.CHECKPOINT_NAME_FORMAT)

    translated_dataset = data_services.load_translated_dataset(
        os.path.join(const.PATH_TO_TRANSLATED_DATASETS, TRANSLATED_DATASET_NAME))
    data_matrix, note2idx, idx2note, dataset_params = translated_dataset
    music_dataset = data_services.create_dataset(
        load_translated_dataset_result=translated_dataset,
        seq_len=SEQ_LEN,
        flat=CREATE_DATASET_FLAT)
    # TODO rather save whole model
    utils.save({const.PM_BATCH_SIZE: BATCH_SIZE,
                const.PM_EMBEDDING_DIM: EMBEDDING_DIM,
                const.PM_RNN_UNITS: RNN_UNITS,
                const.PM_RNN_STATEFUL: RNN_STATEFUL},
               os.path.join(SAVE_MODEL_DIR, const.FN_MODEL_PARAMS))
    created_model = create_model(vocab_size=idx2note.size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS,
                                 rnn_stateful=RNN_STATEFUL, batch_size=BATCH_SIZE)
    train_model(dataset=music_dataset,
                model=created_model,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                load_model_dir=LOAD_MODEL_DIR,
                checkpoint_path=CHECKPOINT_PATH)


if __name__ == '__main__':
    main()
