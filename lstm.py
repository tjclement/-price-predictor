#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KICKSTARTER: Ben
DATE: 10-11 March
"""

# IMPORT OF LIBRARIES
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, RepeatVector, LocallyConnected1D, LeakyReLU, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from test_accuracy import PredictionTester


# CONSTANTS
START_TIME = time.time() # start of total execution time measurement
TRAIN_PROPORTION = 0.7 # training set
VAL_PROPORTION = 0.1 # validations set, hence TEST_PROPORTION = 1 - TRAIN_PROPORTION - VAL_PROPORTION
LOOK_BACK = 1 # hyperparameter
SEED = 0
BATCHES = 1
FEATURE_DIM = 4
OUTPUT_DIM = 1

NEURONS_HIDDEN_LAYER_1 = 10  # hyperparameter
NEURONS_OUTPUT_LAYER = 1
LOSS_FUNCTION = 'mae'  # hyperparameter
OPTIMIZER = 'adamax'  # let's keep it fixed
EPOCHS = 100  # hyperparameter
BATCH_SIZE = 100  # hyperparameter


# FUNCTION DEFINITIONS
def preprocess_dataset(_X, look_back=LOOK_BACK):
    # process dataset so that np.arrays of features and output are extracted
    x, y = [], [] # features (x), output (y)
    y = _X[1:, 3][:-LOOK_BACK]
    for i in range(0, len(_X)-1-LOOK_BACK):
        window = []
        for j in range(LOOK_BACK):
            window.append(_X[i + j, :])
        x.append(window)
    return np.array(x), np.array(y)


def main():
    # fixation of random seed
    np.random.seed(SEED)

    # import of dataset X (change to curated_dataset_2.csv for the second dataset)
    X = pd.read_csv('./data/curated_dataset_1.csv', usecols=[1, 2, 3, 4], engine='python').values \
        .astype('float32')  # raw <np.ndarray>

    # normalization of dataset (recommended for LSTM) (comment out to check numbers easier)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X) # column-wise MinMax scaled np.ndarray

    # split of dataset into training, validation and test set
    train_size = int(len(X) * TRAIN_PROPORTION)
    val_size = int(len(X) * VAL_PROPORTION)
    test_size = len(X) - train_size - val_size
    X_train, X_val, X_test = X[0:train_size, :], X[train_size:train_size+val_size, :], X[train_size+val_size:len(X), :]

    # extraction of features (x) and output (y)
    train_x, train_y = preprocess_dataset(X_train)
    val_x, val_y = preprocess_dataset(X_val)
    test_x, test_y = preprocess_dataset(X_test)

    # reshaping (requirement for Keras input) (samples, timesteps, features)
    train_x = train_x.reshape((len(train_x), LOOK_BACK, FEATURE_DIM))
    val_x = val_x.reshape((len(val_x), LOOK_BACK, FEATURE_DIM))
    test_x = test_x.reshape((len(test_x), LOOK_BACK, FEATURE_DIM))

    # LSTM construction: Input Layer of feature space size, Hidden Layer, Output Layer (topology is hyperparameter)
        # activation: tanh (default: bad), relu (better), None (linear, bad - highly fluctuating)
        # recurrent_activation: hard_sigmoid (default, fine), relu (super bad)
    model = Sequential()
    model.add(LSTM(NEURONS_HIDDEN_LAYER_1, input_shape=( train_x.shape[1], train_x.shape[2]), activation='selu',
                   recurrent_activation='tanh', return_sequences=False))
    model.add(Dense(NEURONS_OUTPUT_LAYER, activation='relu'))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)

    # LSTM fit
    tester = PredictionTester(test_x, test_y, scaler)
    history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), verbose=2,
                        shuffle=False, callbacks=[tester])

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (%s)' % LOSS_FUNCTION)
    plt.legend()
    plt.show() # shows plot of loss function over epochs of training (comment: without normalization it is crap)

    """
    # LSTM evaluation on the test set
    y_predicted = predict(test_x)

    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    """

    print('Script was successfully executed in %.8s s.' % (time.time() - START_TIME))

# SCRIPT EXECUTION
if __name__ == '__main__':
    main()


# END OF FILE