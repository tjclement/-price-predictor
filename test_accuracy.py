"""
Contains some classes to get a grip on the output of the Keras fit() function.
"""

import keras
import matplotlib.pyplot as plt
import math

class PredictionTester(keras.callbacks.Callback):
    features = []
    responses = []
    features_primes = []
    features_train = []
    responses_train = []
    features_primes_train = []
    figure = None
    graph = None
    rmse_train_list = []
    rmse_val_list = []
    mae_train_list = []
    mae_val_list = []

    def __init__(self, features, responses, features_primes, features_train, responses_train, features_primes_train):
        super().__init__()
        self.features = features
        self.responses = responses
        self.features_primes = features_primes
        self.features_train = features_train
        self.responses_train = responses_train
        self.features_primes_train = features_primes_train
        self.figure = plt.figure()
        self.graph = self.figure.add_subplot(111)
        self.rmse_train_list = []
        self.rmse_val_list = []
        self.mae_train_list = []
        self.mae_val_list = []
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        plt.figure()
        plt.plot(self.rmse_train_list, label='rmse_train')
        plt.plot(self.rmse_val_list, label='rmse_val')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE loss [USD]')
        plt.title('RMSE loss on training and validation set over training epochs')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.mae_train_list, label='mae_train')
        plt.plot(self.mae_val_list, label='mae_val')
        plt.xlabel('Epoch')
        plt.ylabel('MAE loss [USD]')
        plt.title('MAE loss on training and validation set over training epochs')
        plt.legend()
        plt.show()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        features_primes = self.features_primes # for back-transforming the price
        predictions = self.model.predict(self.features)
        actual = self.responses # Tom tricked me and I did not see. Shame on me. Let's not flip it.

        # reverse-transform (Real-Use Scenario): element-wise addition before element-wise multiplication
        for i in range(0, len(predictions)):
            predictions[i] = (predictions[i] + 1) * features_primes[i]
            # predictions[i] = predictions[i] * (features_primes[i][1] - features_primes[i][0]) + features_primes[i][0]

        self.graph.clear()
        self.graph.plot(predictions, label='predicted')
        self.graph.plot(actual, label='actual')
        self.graph.set_xlabel('Time [h]')
        self.graph.set_ylabel('Weighted price [USD]')
        self.graph.set_title('Epoch %d (Validation set) (in USD space)' % epoch)
        self.graph.legend()
        self.figure.show()
        plt.pause(0.001)

        # verbose output: train_loss and val_loss
        # predictions, actual already has corrected data from above

        # Repeat like above for validation set on TRAINING SET
        features_primes_train = self.features_primes_train # for back-transforming the price
        predictions_train = self.model.predict(self.features_train)
        actual_train = self.responses_train # Tom tricked me and I did not see. Shame on me. Let's not flip it.

        # reverse-transform (Real-Use Scenario): element-wise addition before element-wise multiplication
        for i in range(0, len(predictions_train)):
            predictions_train[i] = (predictions_train[i] + 1) * features_primes_train[i]

        # RMSE (root mean squared error) FOR TRAIN
        se_train = 0
        for i in range(0, len(predictions_train)):
            se_train += (predictions_train[i]-actual_train[i])**2
        mse_train = se_train / len(predictions_train)
        rmse_train = math.sqrt(mse_train)

        # MAE (mean absolute error) FOR TRAIN
        ae_train = 0
        for i in range(0, len(predictions_train)):
            ae_train += abs(predictions_train[i]-actual_train[i])
        mae_train = ae_train / len(predictions_train)

        # RMSE FOR VAL
        se_val = 0
        for i in range(0, len(predictions)):
            se_val += (predictions[i]-actual[i])**2
        mse_val = se_val / len(predictions)
        rmse_val = math.sqrt(mse_val)

        # MAE (mean absolute error) FOR VAL
        ae_val = 0
        for i in range(0, len(predictions)):
            ae_val += abs(predictions[i]-actual[i])
        mae_val = ae_val / len(predictions_train)

        print('\t in USD - train_rmse: %.4f - val_rmse: %.4f - train_mae: %.4f - val_mae: %.4f \\'
              ' beans_value: 69' % (rmse_train, rmse_val, mae_train, mae_val))

        """
        # If you wanna see the prediction on the training set instead of validation set (then comment out the top graph)
        self.graph.clear()
        self.graph.plot(predictions_train, label='predicted')
        self.graph.plot(actual_train, label='actual')
        self.graph.set_xlabel('Time [h]')
        self.graph.set_ylabel('Weighted price [USD]')
        self.graph.set_title('Epoch %d (Training set) (in USD space)' % epoch)
        self.graph.legend()
        self.figure.show()
        plt.pause(0.001)
        """
        self.rmse_train_list.append(rmse_train)
        self.rmse_val_list.append(rmse_val)
        self.mae_train_list.append(mae_train)
        self.mae_val_list.append(mae_val)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return