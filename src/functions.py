import numpy as np
import pickle

import tensorflow

import config as config


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_data(X, y):
    # Normalize
    X = np.array(X) / 255

    # Reshape the data
    X = X.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, 1)
    y = np.array(y)
    y = tensorflow.keras.utils.to_categorical(y)
    return X, y
