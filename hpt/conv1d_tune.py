import os
import time
import yaml
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils.evaluation import calculate_nse

# --- SET GLOBAL SEEDS FOR REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load config
def load_config():
    with open("hpt/config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()
TRAINING_PARAMS = config["training_params"]
ARCHITECTURAL_PARAMS = config["architectural_params"]["conv1d"]

DATA_PATH_TRAIN = config["paths"]["train"]
DATA_PATH_DEV = config["paths"]["dev"]
RESULTS_DIR = config["paths"]["results"]

features = config["features"]
target = config["target"]

os.makedirs(RESULTS_DIR, exist_ok=True)

def build_conv1d_model(input_shape, num_layers, filters, kernel_size):
    model = Sequential()
    for i in range(num_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=TRAINING_PARAMS["activation"], padding='same', input_shape=input_shape))
        model.add(Dropout(TRAINING_PARAMS["dropout"]))
    model.add(Flatten())
    model.add(Dense(1))
    optimizer = Adam(learning_rate=TRAINING_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss=TRAINING_PARAMS["loss_function"])
    return model

def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def run_architectural_tuning(train_df, dev_df, batch_ids):
    results = []
    for gid in batch_ids:
        train_site = train_df[train_df["gauge_seq_id"] == gid].copy()
        dev_site = dev_df[dev_df["gauge_seq_id"] == gid].copy()
        if train_site.empty or dev_site.empty:
            continue

        X_train = StandardScaler().fit_transform(train_site[features])
        y_train = train_site[target].values.ravel()
        X_dev = StandardScaler().fit_transform(dev_site[features])
        y_dev = dev_site[target].values.ravel()

        for num_layers, filters, kernel_size in product(
            ARCHITECTURAL_PARAMS["num_layers"],
            ARCHITECTURAL_PARAMS["filters"],
            ARCHITECTURAL_PARAMS["kernel_size"]):

            start_time = time.time()
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, TRAINING_PARAMS["lookback"])
            X_dev_seq, y_dev_seq = create_sequences(X_dev, y_dev, TRAINING_PARAMS["lookback"])

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_conv1d_model((X_train_seq.shape[1], X_train_seq.shape[2]), num_layers, filters, kernel_size)
            model.fit(X_train_seq, y_train_seq, epochs=TRAINING_PARAMS["epochs"], batch_size=TRAINING_PARAMS["batch_size"], validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)

            train_nse = calculate_nse(y_train_seq, model.predict(X_train_seq).flatten())
            dev_nse = calculate_nse(y_dev_seq, model.predict(X_dev_seq).flatten())

            results.append({
                "gauge_seq_id": gid,
                "num_layers": num_layers,
                "filters": filters,
                "kernel_size": kernel_size,
                "train_nse": train_nse,
                "dev_nse": dev_nse,
                "nse_difference": train_nse - dev_nse,
                "time_sec": round(time.time() - start_time, 2)
            })

    results_df = pd.DataFrame(results)
    start_seq = batch_ids[0]
    batch_size = len(batch_ids)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"CONV1D_HPT_results_batch_{start_seq}_to_{start_seq + batch_size - 1}.csv"), index=False)

def run_training_param_tuning(train_df, dev_df, batch_ids, tune_param, tune_values):
    results = []
    for gid in batch_ids:
        train_site = train_df[train_df["gauge_seq_id"] == gid].copy()
        dev_site = dev_df[dev_df["gauge_seq_id"] == gid].copy()
        if train_site.empty or dev_site.empty:
            continue

        X_train = StandardScaler().fit_transform(train_site[features])
        y_train = train_site[target].values.ravel()
        X_dev = StandardScaler().fit_transform(dev_site[features])
        y_dev = dev_site[target].values.ravel()

        for value in tune_values:
            start_time = time.time()
            temp_params = TRAINING_PARAMS.copy()
            temp_params[tune_param] = value
            lookback = temp_params["lookback"]

            X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
            X_dev_seq, y_dev_seq = create_sequences(X_dev, y_dev, lookback)

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_conv1d_model((X_train_seq.shape[1], X_train_seq.shape[2]), 2, 64, 5)
            model.fit(X_train_seq, y_train_seq, epochs=temp_params["epochs"], batch_size=temp_params["batch_size"], validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)

            train_nse = calculate_nse(y_train_seq, model.predict(X_train_seq).flatten())
            dev_nse = calculate_nse(y_dev_seq, model.predict(X_dev_seq).flatten())

            results.append({
                "gauge_seq_id": gid,
                tune_param: value,
                "train_nse": train_nse,
                "dev_nse": dev_nse,
                "nse_difference": train_nse - dev_nse,
                "time_sec": round(time.time() - start_time, 2)
            })

    results_df = pd.DataFrame(results)
    start_seq = batch_ids[0]
    batch_size = len(batch_ids)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"CONV1D_HPT_{tune_param}_batch_{start_seq}_to_{start_seq + batch_size - 1}.csv"), index=False)
