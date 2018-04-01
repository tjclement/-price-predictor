#!/usr/bin/env python
# -*- coding: utf-8 -*-

# IMPORT OF LIBRARIES
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import model_from_json
np.seterr(divide='ignore', invalid='ignore')

# FUNCTION DEFINITIONS
def preprocess_dataset(X, look_back, features_num = 1, normalized = True):
    x, x_primes = [], [] # features
    y = X[look_back:, 3]  # output

    for i in range(0, len(X)-look_back):
        window = []
        for j in range(i, i+look_back):
            window.append(X[j, 3])
        top_row = window[0]
        if normalized:
            for i in range(0, len(window)):
                window[i] = (window[i]/top_row)-1  # element-wise division and subtraction with broadcasting
            x_primes.append(top_row)
        x.append(window)
    x, y, x_primes = np.array(x), np.array(y), np.array(x_primes)
    if normalized:
        y = (y/x_primes)-1
        return x, y, x_primes
    else:
        return x, y


def main():
    # DATA SET IMPORT
    X_test = pd.read_csv('./data/curated_dataset_2.csv', usecols=[1, 2, 3, 4]).values \
        .astype('float32') # used for testing
    print('Second data set loaded, contains un-gapped data for %i hours (samples) for 4 features each.' % len(X_test))

    # PRE-PROCESSING OF TRAIN, VAL AND TEST SET
    LOOK_BACK = 6 # [h], 1 month

    X_test = X_test[:2000]

        # TEST
    test_x_norm, test_y_norm, test_x_primes = preprocess_dataset(X_test, look_back=LOOK_BACK, normalized=True)
    test_x, test_y = preprocess_dataset(X_test, look_back=LOOK_BACK, normalized=False)

    # RESHAPING FOR KERAS
    FEATURES_DIM = 1
    test_x_norm = test_x_norm.reshape((len(test_x_norm), LOOK_BACK, FEATURES_DIM))
    test_x = test_x.reshape((len(test_x_norm), LOOK_BACK, FEATURES_DIM))

    # LOADING OF MODEL
    json_file = open('seeds_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("seeds_model.h5")

    # MODEL COMPILATION
    BP_LOSS = 'mean_squared_error'  # back propagation loss
    OPTIMIZER = 'adam'
    model.compile(loss=BP_LOSS, optimizer=OPTIMIZER)

    # MODEL SUMMARY OUTPUT
    model.summary()

    # DOING THE SEED PREDICTIONS
        # DETERMINATION OF THE SEEDS' INITIALIZERS
    hours_predicted = 2 # 10h
    starter_indices = range(10, len(test_x_norm)-1-hours_predicted, 2)

    seed_predictions = []
    for starter_index in starter_indices:
        hours_predicted_for = [starter_index-6]
        predicted_prices = [test_x[starter_index, :, :][0,0]]
        cur_window_norm = test_x_norm[starter_index-1, :, :].reshape((1,LOOK_BACK,1)).copy()
        cur_window_real = test_x[starter_index-1, :, :].reshape((1,LOOK_BACK,1)).copy()
        normalizer = test_x_primes[starter_index-1].copy()
        # print (cur_window_norm) #
        for prediction_hours in range(starter_index, starter_index+hours_predicted):
            hours_predicted_for.append(prediction_hours-5)
            prediction_norm = model.predict(cur_window_norm)
            prediction = ((prediction_norm+1)*normalizer).copy()
            predicted_prices.append(prediction[0,0].copy())
            cur_window_real = cur_window_real[:, 1:, :].copy()
            cur_window_real = np.append(cur_window_real.copy(), np.array([prediction].copy()), axis=1)
            # normalize cur_window_real, to generate the cur_window_norm and the respective normalizer
            cur_window_norm = cur_window_real.copy()
            normalizer = cur_window_norm[:, 0, :].copy()
            for a in range(0, LOOK_BACK):
                cur_window_norm[0, a, 0] = ((cur_window_norm[0, a, 0] / normalizer) - 1).copy()
            # print (cur_window_norm)
        # print (hours_predicted_for)
        # print (predicted_prices)
        seed_predictions.append((hours_predicted_for.copy(), predicted_prices.copy()))
        # print(predicted_prices)

    # calcs done
    print('Calcs Done. Now plotting ...')
    #print (seed_predictions)
    #print(predicted_prices)

    # PLOTTING
    plt.figure()
    lbl_num = 1
    for seed_prediction in seed_predictions:
        a, b = seed_prediction
        hours_predicted_for, predicted_prices = seed_prediction
        plt.plot(hours_predicted_for, predicted_prices, 'C2', label='predictions seed_%i' % lbl_num)
        lbl_num += 1
    plt.plot(test_y, 'C1', label='actual')
    plt.xlabel('Time [h]')
    plt.ylabel('Weighted price [USD]')
    plt.title('Seed predictions on test set vs. actual')
    # plt.legend()
    plt.show()

    print('\nScript was successfully executed.')

# SCRIPT EXECUTION
if __name__ == '__main__':
    main()

# END OF FILE