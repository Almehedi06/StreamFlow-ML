# lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, config):
    model = Sequential()
    num_layers = config.get("num_layers", 1)
    units = config.get("neurons", 32)
    dropout = config.get("dropout", 0.2)
    activation = config.get("activation", "relu")

    for i in range(num_layers):
        return_seq = i < (num_layers - 1)
        if i == 0:
            model.add(LSTM(units=units, activation=activation, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, activation=activation, return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.get(config.get("optimizer", "adam")),
        loss=config.get("loss_function", "mse"),
        metrics=["mae"]
    )
    return model
