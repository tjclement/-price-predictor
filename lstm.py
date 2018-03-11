#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Ben
Date: 10 March
"""

# import of libraries
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

# start of total execution time measurement
START_TIME = time.time()

# constants
TRAIN_PROPORTION = 0.8 # proportion of training set, 1 - TRAIN_PROPORTION == TEST_PROPORTION
LOOK_BACK = 5
SEED = 0

# fixation of random seed
np.random.seed(SEED)

# function definitions


def extract_features_and_outputs(dataset, look_back=LOOK_BACK):
    # process dataset so that np.arrays of features and output are extracted
    x, y = [], [] # features (x), output (y)
    for i in range(0, len(dataset)-look_back):
        dataset_window = dataset[i:(i+look_back), :]
        x.append(dataset_window)
        y.append(dataset[i+look_back, -1])
    return np.array(x), np.array(y)


# import of dataset
dataset = pd.read_csv('./data/curated_dataset_1.csv', usecols=[1,2,3,4], engine='python').values\
    .astype('float32') # raw <np.ndarray>

# normalization of dataset (? should we keep)
dataset = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset) # scaled <np.ndarray>

# split of dataset into training and test dataset
train_size = int(len(dataset) * TRAIN_PROPORTION)
test_size = len(dataset) - train_size
dataset_train, dataset_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# extraction of features (x) and output (y)
train_x, train_y = extract_features_and_outputs(dataset_train)
test_x, test_y = extract_features_and_outputs(dataset_test)

print('Features:')
print(test_x[-1])
print('Output:')
print(test_y[-1])

# next step: reshape so that i can be processed in Keras LSTM

def main():
    pass

if __name__ == '__main__':
    main()

print('Script was successfully executed in %.8s s.' %(time.time() - START_TIME))
# end of script