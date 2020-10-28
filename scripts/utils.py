import os
import pickle
from typing import Any

import tensorflow as tf


def log(log_type, msg):
    if log_type == 'i':
        print('[I] ' + msg)
    elif log_type == 'd':
        print('[D] ' + msg)
    else:
        print('[-] ' + msg)


def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load(path: str) -> Any:
    with open(path, 'rb') as file:
        return pickle.load(file)
