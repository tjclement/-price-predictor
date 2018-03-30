#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stateless LSTM with trunctated BPTT for Stock-Price Prediction (Bitcoin) with a TensorFlow backend
Authors: BW, TC, YD, YK and AS.

This approach adheres to the REAL-USE CASE.
We had a particular view on which data would be available in real use application. For this model only the data (4 features)
for the time of LOOK_BACK hours needs to be available. The forward and reverse_transform (carefully hand-crafted) are
only applied to this LOOK_BACK window. No normalization based on the whole data set is used, although used in many implementations
online, but seen as inaccurate by us.

The approach could be manipulated, so to be used for other stock price predictions.

Used window-normalization (example with LOOK_BACK = 2):
    feature1 (min)    feature2 (max)    feature3 (vol)    feature4 (price)         Y
    1                   1               1                   2                      20       this row is stored for each window*
    2                   2               2                   10

    forward-transform (used to train the LSTM model)
    x_(row=a, col=b) = (x_(row=a, col=b) / x_(row=0, col=b)) - 1
    gives following window-normalized:
    feature1 (min)    feature2 (max)    feature3 (vol)    feature4 (price)         Y         prediction by LSTM is done on this data
    0                   0               0                   0                      20/2-1=9  (also internal back-propagation)
    1                   1               1                   4

    reverse_transform of prediction (uses row stored above*) done by LSTM (to calculate evaluation metrics):
        x_(row=a, col=b) = (x_(row=a, col=b) + 1) * x_(row=0, col=b)
"""

print('Good afternoon, gentlemen!')

# IMPORT OF LIBRARIES
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from test_accuracy import PredictionTester


# CONSTANTS
START_TIME = time.time()
NEURONS_HIDDEN_LAYER_1 = 9
NEURONS_OUTPUT_LAYER = 1
FEATURES_NUM = 4

# FUNCTION DEFINITIONS
def preprocess_dataset(X, look_back, features_num = FEATURES_NUM, normalized = True):
    x, x_primes = [], [] # features
    y = X[look_back:, 3]  # output

    for i in range(0, len(X)-look_back):
        window = []
        for j in range(i, i+look_back):
            window.append(X[j, :])
        top_row = window[0]
        if normalized:
            first_row = window[0]
            for i in range(0, len(window)):
                window[i] = (window[i]/top_row)-1  # element-wise division and subtraction with broadcasting
            x_primes.append(first_row[3])
        x.append(window)
    x, y, x_primes = np.array(x), np.array(y), np.array(x_primes)
    if normalized:
        y = (y/x_primes)-1
        return x, y, x_primes
    else:
        return x, y


def main():
    # INITIAL ASSIGNMENT OF RANDOM SEED FOR REPRODUCIBILITY
    SEED = 0
    np.random.seed(SEED)

    # DATA SET IMPORT
    X = pd.read_csv('./data/curated_dataset_1.csv', usecols=[1, 2, 3, 4]).values \
        .astype('float32') # used for training and validaton
    print('First data set loaded, contains un-gapped data for %i hours (samples) for 4 features each.' % len(X))
    X_additional = pd.read_csv('./data/curated_dataset_2.csv', usecols=[1, 2, 3, 4]).values \
        .astype('float32') # used for testing
    print('Second data set loaded, contains un-gapped data for %i hours (samples) for 4 features each.' % len(X_additional))

    # SPLIT FIRST DATA SET INTO TRAINING AND VALIDATION
    VAL_SIZE = 24*7*20 # 4 months
    train_size = len(X)-VAL_SIZE # -TEST_SIZE
    X_train = X[0:train_size, :]  # first in time dimension
    X_val = X[train_size:, :] # second in time dimension

    # MAKE SECOND DATASET THE TEST SET
    X_test = X_additional  # third in time dimension with 1 year split to first part

    # PRE-PROCESSING OF TRAIN, VAL AND TEST SET
    LOOK_BACK = 24*7 # [h], so 7 days, 1 week

        # TRAIN
    train_x_norm, train_y_norm, train_x_primes = preprocess_dataset(X_train, look_back=LOOK_BACK, normalized=True)
    train_x, train_y = preprocess_dataset(X_train, look_back=LOOK_BACK, normalized=False)

        # VAL
    val_x_norm, val_y_norm, val_x_primes = preprocess_dataset(X_val, look_back=LOOK_BACK, normalized=True)
    val_x, val_y = preprocess_dataset(X_val, look_back=LOOK_BACK, normalized=False)

        # TEST
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
        # TERMINAL DENSE LAYER DELIVERING TO 1 OUTPUT NODE
    model.add(Dense(NEURONS_OUTPUT_LAYER))

    # MODEL COMPILATION
    BP_LOSS = 'mean_squared_error'  # back propagation loss
    OPTIMIZER = 'adam'
        # Adam - A Method for Stochastic Optimization (http://arxiv.org/abs/1412.6980v8)
        # On the Convergence of Adam and Beyond (https://openreview.net/forum?id=ryQu7f-RZ)
    model.compile(loss=BP_LOSS, optimizer=OPTIMIZER)

    # MODEL SUMMARY OUTPUT
    model.summary()

    # FITTING OF MODEL ONTO TRAINING DATA
    EPOCHS = 100
    BATCH_SIZE = 100
    tester = PredictionTester(val_x_norm, val_y, val_x_primes, train_x_norm, train_y, train_x_primes) # see: test_accuracy.py module
    model.fit(x=train_x_norm, y=train_y_norm, shuffle=True, validation_data=(val_x_norm, val_y_norm), epochs=EPOCHS,
              batch_size=BATCH_SIZE, verbose=2, callbacks=[tester])
    print('Training finished.')

    """"
    # SAVE MODEL (including WEIGHTS) FOR LATER USE (CURRENTLY JUST BACK-UP)
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # you require H5PY package: pip install h5py
    SAVE A MODEL BY:
            # serialize model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")
    LOAD MODEL BY:
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            print("Loaded model from disk")
            loaded_model.compile(...) # new model name: loaded_model
    """
    # SAVE MODEL
        # this causes error: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
        # from ._conv import register_converters as _register_converters (WHICH CAN BE IGNORED, read online)
        # SAVE TOPOLOGY TO JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # SAVE WEIGHTS TO HDF5
    model.save_weights("model.h5")
    print("Saved model to disk (carefully crafted weights and topology are save)")

    """
    # FINAL EVALUATION ON TEST SET
    test_predictions = model.predict(test_x_norm)
    for i in range(0, len(test_predictions)):
        test_predictions[i] = (test_predictions[i] + 1) * test_x_primes[i]
    print ('------------------------ TEST SET ------------------------')

        # RMSE (root mean squared error) FOR TEST
    se_test = 0
    for i in range(0, len(test_predictions)):
        se_test += (test_predictions[i] - test_y[i]) ** 2
    mse_test = se_test / len(test_predictions)
    rmse_test = math.sqrt(mse_test)

        # MAE (mean absolute error) FOR TEST
    ae_test = 0
    for i in range(0, len(test_predictions)):
        ae_test += abs(test_predictions[i] - test_y[i])
    mae_test = ae_test / len(test_predictions)
    print('Final test on TEST set (size: %i) was done with rmse_test: %.4f - mae_test %.4f' % (len(test_predictions), rmse_test, mae_test))

        # TEST PLOT
    plt.figure()
    plt.plot(test_predictions, label='predictions')
    plt.plot(test_y, label='actual')
    plt.xlabel('Time [h]')
    plt.ylabel('Weighted price [USD]')
    plt.title('Prediction on test set vs. actual')
    plt.legend()
    plt.show()
    """

    print('\nScript was successfully executed in %.8s s.' % (time.time() - START_TIME))

# SCRIPT EXECUTION
if __name__ == '__main__':
    main()

# END OF FILE