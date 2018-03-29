from keras import Sequential
from keras.layers import LSTM, Dense

def LSTMModel(neurons, look_back, features_dim, output):
    """
    BUILDING OF STATELESS (stateful=False) LSTM WITH TRUNCATED BPTT (GIVEN BY LOOK_BACK) WITH SEQUENTIAL MODEL
    :param neurons:
    :param look_back:
    :param features_dim:
    :param output:
    :return:
    """
    model = Sequential()
    # LSTM LAYER
    model.add(LSTM(neurons, input_shape=(look_back, features_dim), return_sequences=False))
    # TERMINAL DENSE LAYER DELIVERING TO 1 OUTPUT NODE
    model.add(Dense(output))

    return model