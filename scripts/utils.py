import os
import pickle
from typing import Any

import numpy as np
import tensorflow as tf

import constants as const


def log(log_type, msg):
    if log_type == 'i':
        print('[I] ' + msg)
    elif log_type == 'd':
        print('[D] ' + msg)
    elif log_type == 'w':
        print('[W] ' + msg)
    else:
        print('[-] ' + msg)


def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


def example_generator(sequence: np.ndarray, seq_len: int):
    idx_low = range(0, sequence.size - seq_len, seq_len)
    idx_high = range(seq_len, sequence.size, seq_len)
    for ilo, ihi in zip(idx_low, idx_high):
        yield sequence[ilo:ihi], sequence[ilo + 1:ihi + 1]


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load(path: str) -> Any:
    with open(path, 'rb') as file:
        return pickle.load(file)


def verify_training_params(model_name: str, params_dict: dict, warn_only=False):
    try:
        saved_params = load(os.path.join(const.PATH_TO_CHECKPOINTS, model_name, const.FN_MODEL_PARAMS))
    except FileNotFoundError:
        return

    for key in saved_params.keys():
        if params_dict[key] != saved_params[key]:
            msg = 'Param %s differs from saved param (model: %s)' % (key, model_name)
            if warn_only:
                log('w', msg)
            else:
                raise ValueError(msg + '\nAborting execution...')


if __name__ == '__main__':
    # log('d', '(09, 5):' + str(list(example_generator(np.arange(9), 5))))
    # log('d', '(10, 5):' + str(list(example_generator(np.arange(10), 5))))
    # log('d', '(11, 5):' + str(list(example_generator(np.arange(11), 5))))
    # log('d', '(12, 5):' + str(list(example_generator(np.arange(12), 5))))
    # log('d', '(13, 5):' + str(list(example_generator(np.arange(13), 5))))
    # log('d', '(14, 5):' + str(list(example_generator(np.arange(14), 5))))
    # log('d', '(15, 5):' + str(list(example_generator(np.arange(15), 5))))
    # log('d', '(16, 5):' + str(list(example_generator(np.arange(16), 5))))
    # log('d', '(17, 5):' + str(list(example_generator(np.arange(17), 5))))
    pass
