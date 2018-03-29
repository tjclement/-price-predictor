from keras import Sequential
from keras.layers import Dense, Flatten


def LinearRegressionModel(look_back, features_dim, output):
    """
    Simple linear regressor, implemented in single layer neural network
    :param look_back:
    :param features_dim:
    :param output:
    :return:
    """
    model = Sequential()
    model.add(Flatten(input_shape=(look_back, features_dim)))
    model.add(Dense(output, init='uniform', activation='linear'))
    return model