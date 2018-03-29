from keras import Sequential
from keras.layers import LSTM, Dense

def TomModel(look_back, features_dim, output):
    """
    :param look_back:
    :param features_dim:
    :param output:
    :return:
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, features_dim), return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))

    return model