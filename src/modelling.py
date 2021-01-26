import numpy as np
import pickle

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from functions import load_data, preprocess_data
import config as config

import logging
import os


if not os.path.exists('../log'):
    os.makedirs('../log')

logging.basicConfig(
    filename='../log/modelling.log',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)


# Load the data
X_train = load_data('../data/processed_data/X_train.pkl')
y_train = load_data('../data/processed_data/y_train.pkl')
X_val = load_data('../data/processed_data/X_val.pkl')
y_val = load_data('../data/processed_data/y_val.pkl')

# Preprocess the data
X_train, y_train = preprocess_data(X_train, y_train)
X_val, y_val = preprocess_data(X_val, y_val)

# Build the model
logging.info(f'The reshaped image size is {config.IMG_SIZE} by {config.IMG_SIZE}')
input_shape = (config.IMG_SIZE, config.IMG_SIZE, 1)
model = Sequential()

logging.info(f'The size of the input layer is {config.INPUT_LAYER_SIZE}')
logging.info(f'The size of the kernel is {config.KERNEL_SIZE}')
logging.info(f'The size of the strides in conv layer is {config.CONV_STRIDES}')
logging.info(f'The activation is {config.ACTIVATION}')

model.add(Conv2D(
    config.INPUT_LAYER_SIZE,
    kernel_size=config.KERNEL_SIZE,
    strides=config.CONV_STRIDES,
    activation=config.ACTIVATION,
    input_shape=input_shape
))

logging.info(f'The size of the hidden layer is {config.HIDDEN_LAYER_SIZE}')
logging.info(f'The dropout rate is {config.DROPOUT}')
logging.info(f'The pool size is {config.POOL_SIZE}')
logging.info(f'The strides in the maxpooling layer is {config.POOL_STRIDES}')

model.add(Dropout(config.DROPOUT))
for _ in range(config.HIDDEN_LAYER - 1):
    model.add(MaxPooling2D(pool_size=config.POOL_SIZE, strides=config.POOL_STRIDES))
    model.add(Conv2D(
        config.HIDDEN_LAYER_SIZE,
        kernel_size=config.KERNEL_SIZE,
        strides=config.CONV_STRIDES,
        activation=config.ACTIVATION
    ))
    model.add(Dropout(config.DROPOUT))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

logging.info(f'The loss function is {config.LOSS}')
logging.info(f'The optimizer is {config.OPTIMIZER}')
logging.info(f'The metrics: {config.METRICS}')

model.compile(
    loss=config.LOSS,
    optimizer=config.OPTIMIZER,
    metrics=config.METRICS
)

logging.info(f'The batch size is {config.BATCH_SIZE}')
logging.info(f'The epochs: {config.EPOCHS}')

history = model.fit(
    X_train,
    y_train,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_data=(X_val, y_val),
    verbose=2
)

logging.info(f'The training accuracy is {round(history.history["acc"][-1], 4)}')
logging.info(f'The validation accuracy is {round(history.history["val_acc"][-1], 4)}')

if not os.path.exists('../output'):
    os.makedirs('../output')

model.save('../output/trained_model')

with open('../output/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
