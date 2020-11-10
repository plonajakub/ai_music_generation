import glob
import os
import sys
import time

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


@tf.function
def train_step(model, optimizer, inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(utils.loss(labels=target, logits=predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def restore_model(model, load_model_dir):
    if len(glob.glob(os.path.join(load_model_dir, const.CHECKPOINT_NAME_GLOB))) != 0:
        model.load_weights(tf.train.latest_checkpoint(load_model_dir))
        log('i', 'Model weights have been loaded from \'' + load_model_dir + '\'')
    else:
        log('i', 'Path \'' + load_model_dir + '\' does not contain checkpoints.')
        log('i', 'Compiling new model...')
    model.compile(optimizer='adam', loss=utils.loss)


def train_model_sparse(dataset, model, batch_size, epochs, load_model_dir, save_model_dir, checkpoint_path):
    for i in range(len(dataset)):
        dataset[i] = dataset[i].batch(batch_size, drop_remainder=True)

    restore_model(model, load_model_dir)

    loss = sys.float_info.max
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        start = time.time()
        batch_n = 0
        for partial_dataset in dataset:
            model.reset_states()
            for inp, target in partial_dataset:
                loss = train_step(model=model, optimizer=optimizer, inp=inp, target=target)

                if batch_n % const.TRAIN_REPORT_FREQ == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch_n + 1, loss))
                batch_n += 1

        try:
            saved_model_loss = utils.load(os.path.join(load_model_dir, const.FN_MODEL_LOSS))
            new_model = False
        except FileNotFoundError:
            new_model = True

        if new_model or loss < saved_model_loss:
            model.save_weights(checkpoint_path)
            utils.save(loss, os.path.join(save_model_dir, const.FN_MODEL_LOSS))
            print('New weights saved! Loss {:.4f}'.format(loss))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def train_model_flat(dataset, model, batch_size, epochs, load_model_dir, checkpoint_path):
    dataset = dataset.batch(batch_size, drop_remainder=True)
    restore_model(model, load_model_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True)

    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


def main():
    # Data
    TRANSLATED_DATASET_NAME = 'bach_fugue_all_timing_true'  # Load
    CREATE_DATASET_FLAT = False
    SEQ_LEN = 500

    # Model creation
    BATCH_SIZE = 1  # Batch size = 1 would be good for stateful = True
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    RNN_STATEFUL = True

    # Training
    EPOCHS = 100
    MODEL_NAME = 'bach_fugue_stateful_true_train_sparse_dataset'  # Save
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
    if CREATE_DATASET_FLAT:
        train_model_flat(dataset=music_dataset,
                         model=created_model,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         load_model_dir=LOAD_MODEL_DIR,
                         checkpoint_path=CHECKPOINT_PATH)
    else:
        train_model_sparse(dataset=music_dataset,
                           model=created_model,
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           load_model_dir=LOAD_MODEL_DIR,
                           save_model_dir=SAVE_MODEL_DIR,
                           checkpoint_path=CHECKPOINT_PATH)


if __name__ == '__main__':
    main()
