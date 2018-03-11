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
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# CONSTANTS
START_TIME = time.time() # start of total execution time measurement
TRAIN_PROPORTION = 0.8 # proportion of training set, 1 - TRAIN_PROPORTION == TEST_PROPORTION
LOOK_BACK = 5 # hyperparameter
SEED = 0
BATCHES = 1
FEATURE_DIM = 4
OUTPUT_DIM = 1


# FUNCTION DEFINITIONS
def preprocess_dataset(_X, look_back=LOOK_BACK):
    # process dataset so that np.arrays of features and output are extracted
    x, y = [], [] # features (x), output (y)
    y = _X[1:, 3]
    for i in range(0, len(_X)-1):
         x.append(_X[i, :])
    return np.array(x[:len(x) - (len(x) % look_back)]), np.array(y[:len(y) - (len(y) % look_back)])


def main():
    # fixation of random seed
    np.random.seed(SEED)

    # import of dataset X (change to curated_dataset_2.csv for the second dataset)
    X = pd.read_csv('./data/curated_dataset_1.csv', usecols=[1, 2, 3, 4], engine='python').values \
        .astype('float32')  # raw <np.ndarray>

    # normalization of dataset (recommended for LSTM) (comment out to check numbers easier)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X) # scaled <np.ndarray>

    # split of dataset into training and test dataset
    train_size = int(len(X) * TRAIN_PROPORTION)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]

    # extraction of features (x) and output (y)
    train_x, train_y = preprocess_dataset(X_train)
    test_x, test_y = preprocess_dataset(X_test)

    # reshaping (requirement for Keras input)
    train_x = train_x.reshape((len(train_x), BATCHES, FEATURE_DIM))
    test_x = test_x.reshape((len(test_x), BATCHES, FEATURE_DIM))

    # LSTM construction
    NEURONS_HIDDEN_LAYER_1 = 50 # hyperparameter
    NEURONS_OUTPUT_LAYER = 1
    LOSS_FUNCTION = 'mae' # hyperparameter
    OPTIMIZER = 'adam' # let's keep it fixed
    EPOCHS = 100 # hyperparameter
    BATCH_SIZE = 50 # hyperparameter

    model = Sequential()
    model.add(LSTM(NEURONS_HIDDEN_LAYER_1, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(NEURONS_OUTPUT_LAYER))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)

    # LSTM fit: here test_x and test_y where used, although they are used as VALIDATION sets in this case
    history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # correction: segregate data into TRAIN (60%), VALIDATION (20%) and TEST (20%) set (roll-forward); (do in beginning)

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (%s)' % LOSS_FUNCTION)
    plt.legend()

    print('Script was successfully executed in %.8s s.' % (time.time() - START_TIME))

    plt.show()

# SCRIPT EXECUTION
if __name__ == '__main__':
    main()


# END OF FILE