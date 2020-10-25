import os

import tensorflow as tf

import data_services
import utils
from utils import log

# Data
PATH_TO_DATA = '..\\data\\bach_all_not_corrupted\\bach_all_not_corrupted_data'
DATA_BLOB = '01[ac]*.mid'
DATASET_WITH_TIMING = True
DATASET_FLAT = True
SEQ_LEN = 100

# Model creation
BATCH_SIZE = 3  # Batch size = 1 would be good for stateful = True (?)
EMBEDDING_DIM = 256
RNN_UNITS = 128
RNN_STATEFUL = True

# Training
EPOCHS = 3
CHECKPOINTS_DIR = '..\\training_checkpoints'
SAVE_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, 'tests')
LOAD_MODEL_DIR = SAVE_MODEL_DIR
CHECKPOINT_PREFIX = os.path.join(SAVE_MODEL_DIR, 'ckpt_{epoch}')


def create_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=RNN_STATEFUL),
        tf.keras.layers.Dense(vocab_size)
    ])
    model.summary()
    return model


def train_model(dataset, model):
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    if os.path.exists(LOAD_MODEL_DIR):
        model.load_weights(tf.train.latest_checkpoint(LOAD_MODEL_DIR))
        log('i', 'Model weights have been loaded from \'' + LOAD_MODEL_DIR + '\'')
    else:
        log('i', 'Path \'' + LOAD_MODEL_DIR + '\' does not contain checkpoints.')
        log('i', 'Compiling new model...')
    model.compile(optimizer='adam', loss=utils.loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PREFIX,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True)

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


if __name__ == '__main__':
    music_dataset, idx2note = data_services.get_music_dataset(data_path=os.path.join(PATH_TO_DATA, DATA_BLOB),
                                                              with_timing=DATASET_WITH_TIMING,
                                                              seq_len=SEQ_LEN, flat=DATASET_FLAT)
    created_model = create_model(vocab_size=idx2note.size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS,
                                 batch_size=BATCH_SIZE)
    train_model(dataset=music_dataset, model=created_model)
