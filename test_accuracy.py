import keras
import numpy as np
import matplotlib.pyplot as plt

class PredictionTester(keras.callbacks.Callback):
    features = []
    responses = []
    features_primes = []
    figure = None
    graph = None

    def __init__(self, features, responses, features_primes):
        super().__init__()
        self.features = features
        self.responses = responses
        self.features_primes = features_primes
        self.figure = plt.figure()
        self.graph = self.figure.add_subplot(111)
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        features_primes = self.features_primes # for back-transforming the price
        predictions = self.model.predict(self.features, batch_size=100)
        actual = self.responses # Tom tricked me and I did not see. Shame on me. Let's not flip it.

        # reverse-transform (Real-Use Scenario): element-wise addition before element-wise multiplication
        for i in range(0, len(predictions)):
            predictions[i] = predictions[i] * (features_primes[i][1] - features_primes[i][0]) + features_primes[i][0]

        self.graph.clear()
        self.graph.plot(predictions, label='predicted')
        self.graph.plot(actual, label='actual')
        self.graph.set_xlabel('Time [h]')
        self.graph.set_ylabel('Weighted price [USD]')
        self.graph.set_title('Epoch %d' % epoch)
        self.graph.legend()
        self.figure.show()
        plt.pause(0.001)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return