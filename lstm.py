#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stateless LSTM with trunctated BPTT for Stock-Price Prediction with a TensorFlow backend
Authors: BW, TC, YD, YK and AS.
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
# VAL_PROPORTION = 0.1 # validation set, hence TEST_PROPORTION = 1 - TRAIN_PROPORTION - VAL_PROPORTION
# VAL_PROPORTION = 0.1 # validations set, hence TEST_PROPORTION = 1 - TRAIN_PROPORTION - VAL_PROPORTION

NEURONS_HIDDEN_LAYER_1 = 9  # HP
NEURONS_OUTPUT_LAYER = 1
FEATURES_NUM = 4

# FUNCTION DEFINITIONS
def preprocess_dataset(X, look_back, features_num = FEATURES_NUM, normalized = True):
    # process dataset so that np.arrays of features and output are extracted
    x, x_primes = [], [] # features (x)
    y = X[1:, 3][look_back:] # output (y)
    y = X[look_back:, 3]  # output (y)

    for i in range(0, len(X)-look_back):
        window = []
        for j in range(i, i+look_back):
            # if i == j:
            # x_primes.append(X[j,features_num-1])
            window.append(X[j, :])
        top_row = window[0]
        if normalized:
            cur_mins, cur_maxs = [], []
            #for i in range(0, len(window)): # applies element-wise division and subtraction with broadcasting on np.arrays
            #    window[i] = (window[i]/top_row)-1
            for a in range(0, features_num):
                window = np.array(window)
                cur_mins.append(min(window[:, a]))
                cur_maxs.append(max(window[:, a]))
            x_primes.append([cur_mins[3], cur_maxs[3]])

            for a in range(0, features_num):
                window[:, a] = (window[:, a] - np.array(cur_mins[a])) / (np.array(cur_maxs[a]) - np.array(cur_mins[a]))

        x.append(window)
    x, y, x_primes = np.array(x), np.array(y), np.array(x_primes)
    if normalized:
        # y = (y/x_primes)-1
        for i in range(0, len(y)):
            y[i] = (y[i] - x_primes[i][0]) / (x_primes[i][1] - x_primes[i][0])
        return x, y, x_primes
    else:
        return x, y

def main():
    # INITIAL ASSIGNMENT OF RANDOM SEED FOR REPRODUCIBILITY
    SEED = 0
    np.random.seed(SEED)

    # DATA SET IMPORT
    X = pd.read_csv('./data/curated_dataset_1.csv', usecols=[1, 2, 3, 4]).values \
        .astype('float32')
        #(change to curated_dataset_2.csv for the second data set available in './data')
        # optionally this could be an additional test set

    # SPLIT INTO TRAINING, VALIDATION AND TEST SET (TEST SET NOT TOUCHED ANYMORE UNTIL THE END FROM HERE ON IN TRAINING)
    VAL_SIZE = 1000
    TEST_SIZE = 5000
    train_size = len(X)-VAL_SIZE-TEST_SIZE
    X_train = X[0:train_size, :]  # first in time dimension
    X_val = X[train_size:train_size+VAL_SIZE, :]
    X_test = X[train_size+VAL_SIZE:, :]

    # PRE-PROCESSING OF TRAIN, VAL AND TEST SET
    LOOK_BACK = 24*7 # [h], so 7 days, 1 week
    train_x_norm, train_y_norm, train_x_primes = preprocess_dataset(X_train, look_back=LOOK_BACK, normalized=True)
    val_x_norm, val_y_norm, val_x_primes = preprocess_dataset(X_val, look_back=LOOK_BACK, normalized=True)
        # test set normalization for input into the prediction (this is data available in the given real-use case)
        # this included just that for a given window the minimum and maximum is known, which will always the be case
    val_x, val_y = preprocess_dataset(X_val, look_back=LOOK_BACK, normalized=False)
    test_x_norm, test_y_norm, test_x_primes = preprocess_dataset(X_test, look_back=LOOK_BACK, normalized=True)
    test_x, test_y = preprocess_dataset(X_test, look_back=LOOK_BACK, normalized=False)

    # RESHAPING FOR KERAS: required dimensions are number_of_samples * time_steps (look_back_number) * features_number
    FEATURES_DIM = 4
    train_x_norm = train_x_norm.reshape((len(train_x_norm), LOOK_BACK, FEATURES_DIM))
    val_x_norm = val_x_norm.reshape((len(val_x_norm), LOOK_BACK, FEATURES_DIM))
    test_x_norm = test_x_norm.reshape((len(test_x_norm), LOOK_BACK, FEATURES_DIM))

    # BUILDING OF STATELESS (stateful=False) LSTM WITH TRUNCATED BPTT (GIVEN BY LOOK_BACK) WITH SEQUENTIAL MODEL
    model = Sequential()
        # LSTM LAYER
    model.add(LSTM(NEURONS_HIDDEN_LAYER_1, input_shape=(LOOK_BACK, FEATURES_DIM), return_sequences=False))
            # explore arguments: activation = 'selu', recurrent_activation = 'tanh'
        # TERMINAL DENSE LAYER DELIVERING TO 1 OUTPUT NODE
    model.add(Dense(NEURONS_OUTPUT_LAYER))
            # explore arguments: activation='relu'

    # MODEL COMPILATION
    BP_LOSS = 'mean_squared_error'  # back propagation loss
    OPTIMIZER = 'adam'
        # Adam - A Method for Stochastic Optimization (http://arxiv.org/abs/1412.6980v8)
        # On the Convergence of Adam and Beyond (https://openreview.net/forum?id=ryQu7f-RZ)
    model.compile(loss=BP_LOSS,optimizer=OPTIMIZER)

    # MODEL SUMMARY OUTPUT
    model.summary()

    # FITTING OF MODEL ONTO TRAINING DATA
    EPOCHS = 50
    BATCH_SIZE = 100
    tester = PredictionTester(val_x_norm, val_y, val_x_primes) # see: test_accuracy.py module
    history = model.fit(x=train_x_norm, y=train_y_norm, validation_data=(val_x_norm, val_y_norm), shuffle=True,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[tester])

    # PLOT HISTORY OF TRAINING AND VALIDATION LOSS OVER THE EPOCHS
    OUTPUT_LOSS = 'mae'
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (%s)' % OUTPUT_LOSS.upper())
    plt.legend()
    plt.show()

    # FINAL EVALUATION ON TEST SET
    # loss_and_metrics = model.evaluate(test_x_norm, test_y_norm, batch_size=BATCH_SIZE)

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