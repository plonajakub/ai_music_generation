import glob
import os

import tensorflow as tf

import data_services
import utils
from utils import log

# Data
PATH_TO_TRANSLATED_DATA = '..\\data\\pickles'
DATASET_NAME = 'bach_sample_var'
DATASET_FLAT = True
SEQ_LEN = 100

# Model creation
BATCH_SIZE = 3  # Batch size = 1 would be good for stateful = True (?)
EMBEDDING_DIM = 256
RNN_UNITS = 128
RNN_STATEFUL = True

# Training
EPOCHS = 20
CHECKPOINTS_DIR = '..\\training_checkpoints'
MODEL_NAME = 'bach_var'
SAVE_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, MODEL_NAME)
LOAD_MODEL_DIR = SAVE_MODEL_DIR
CHECKPOINT_NAME_PATTERN = 'ckpt_[0-9]*'
CHECKPOINT_NAME_FORMAT = 'ckpt_{epoch}'
CHECKPOINT_PATH = os.path.join(SAVE_MODEL_DIR, CHECKPOINT_NAME_FORMAT)


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

    if len(glob.glob(os.path.join(LOAD_MODEL_DIR, CHECKPOINT_NAME_PATTERN))) != 0:
        model.load_weights(tf.train.latest_checkpoint(LOAD_MODEL_DIR))
        log('i', 'Model weights have been loaded from \'' + LOAD_MODEL_DIR + '\'')
    else:
        log('i', 'Path \'' + LOAD_MODEL_DIR + '\' does not contain checkpoints.')
        log('i', 'Compiling new model...')
    model.compile(optimizer='adam', loss=utils.loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True)

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


if __name__ == '__main__':
    translated_dataset = data_services.load_translated_dataset(os.path.join(PATH_TO_TRANSLATED_DATA, DATASET_NAME))
    data_matrix, note2idx, idx2note, dataset_params = translated_dataset
    music_dataset = data_services.create_dataset(
        load_translated_dataset_result=translated_dataset,
        seq_len=SEQ_LEN,
        flat=DATASET_FLAT)
    # TODO rather save whole model
    utils.save({'BATCH_SIZE': BATCH_SIZE, 'EMBEDDING_DIM': EMBEDDING_DIM,
                'RNN_UNITS': RNN_UNITS, 'RNN_STATEFUL': RNN_STATEFUL},
               os.path.join(SAVE_MODEL_DIR, 'model_params.pickle'))
    created_model = create_model(vocab_size=idx2note.size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS,
                                 batch_size=BATCH_SIZE)
    train_model(dataset=music_dataset, model=created_model)
