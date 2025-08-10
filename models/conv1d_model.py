import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense

def build_conv1d_model(input_shape, config):
    """
    Build and compile a Conv1D model using hyperparameters from config.

    Parameters:
    - input_shape: tuple (timesteps, features)
    - config: dict from config.yaml (e.g., config['conv1d'])

    Returns:
    - Compiled Keras Conv1D model
    """
    model = Sequential()

    num_layers = config.get("num_layers", 2)
    filters = config.get("filters", 64)
    kernel_size = config.get("kernel_size", 5)
    dropout = config.get("dropout", 0.2)
    activation = config.get("activation", "relu")

    for i in range(num_layers):
        if i == 0:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation,
                             padding="same", input_shape=input_shape))
        else:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding="same"))
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(
        optimizer=config.get("optimizer", "adam"),
        loss=config.get("loss_function", "mse"),
        metrics=["mae"]
    )

    return model
