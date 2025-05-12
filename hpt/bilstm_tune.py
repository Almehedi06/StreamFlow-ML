# bilstm_tune.py

import os
import time
import yaml
import pandas as pd
import numpy as np
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils.data_loader import split_and_scale, reshape_for_sequence
from utils.evaluation import calculate_nse
import random
import tensorflow as tf

# --- SET GLOBAL SEEDS FOR REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load config
with open("hpt/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRAINING_PARAMS = config["training_params"]
ARCHITECTURAL_PARAMS = config["architectural_params"]["bilstm"]

DATA_PATH_TRAIN = config["paths"]["train"]
DATA_PATH_DEV = config["paths"]["dev"]
RESULTS_DIR = config["paths"]["results"]

features = config["features"]
target = config["target"]

os.makedirs(RESULTS_DIR, exist_ok=True)

def build_bilstm_model(input_shape, num_layers, neurons):
    model = Sequential()
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        if i == 0:
            model.add(Bidirectional(LSTM(neurons, activation=TRAINING_PARAMS["activation"], return_sequences=return_seq), input_shape=input_shape))
        else:
            model.add(Bidirectional(LSTM(neurons, activation=TRAINING_PARAMS["activation"], return_sequences=return_seq)))
        model.add(Dropout(TRAINING_PARAMS["dropout"]))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=TRAINING_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss=TRAINING_PARAMS["loss_function"])
    return model

def run_architectural_tuning(train_df, dev_df, batch_ids):
    results = []
    for gid in batch_ids:
        train_site = train_df[train_df["gauge_seq_id"] == gid].copy()
        dev_site = dev_df[dev_df["gauge_seq_id"] == gid].copy()
        if train_site.empty or dev_site.empty:
            continue

        X_train, y_train = split_and_scale(train_site, features, target)
        X_dev, y_dev = split_and_scale(dev_site, features, target)

        for num_layers, neurons in product(ARCHITECTURAL_PARAMS["num_layers"], ARCHITECTURAL_PARAMS["neurons"]):
            start_time = time.time()
            X_train_seq = reshape_for_sequence(X_train, TRAINING_PARAMS["lookback"])
            y_train_seq = y_train[TRAINING_PARAMS["lookback"]:]
            X_dev_seq = reshape_for_sequence(X_dev, TRAINING_PARAMS["lookback"])
            y_dev_seq = y_dev[TRAINING_PARAMS["lookback"]:]

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_bilstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), num_layers, neurons)
            model.fit(X_train_seq, y_train_seq, epochs=TRAINING_PARAMS["epochs"], batch_size=TRAINING_PARAMS["batch_size"], validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)

            train_nse = calculate_nse(y_train_seq, model.predict(X_train_seq).flatten())
            dev_nse = calculate_nse(y_dev_seq, model.predict(X_dev_seq).flatten())

            results.append({
                "gauge_seq_id": gid,
                "num_layers": num_layers,
                "neurons": neurons,
                "train_nse": train_nse,
                "dev_nse": dev_nse,
                "nse_difference": train_nse - dev_nse,
                "time_sec": round(time.time() - start_time, 2)
            })

    results_df = pd.DataFrame(results)
    start_seq = batch_ids[0]
    batch_size = len(batch_ids)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"bilstm_HPT_results_batch_{start_seq}_to_{start_seq + batch_size - 1}.csv"), index=False)

def run_training_param_tuning(train_df, dev_df, batch_ids, tune_param, tune_values):
    results = []
    for gid in batch_ids:
        train_site = train_df[train_df["gauge_seq_id"] == gid].copy()
        dev_site = dev_df[dev_df["gauge_seq_id"] == gid].copy()
        if train_site.empty or dev_site.empty:
            continue

        X_train, y_train = split_and_scale(train_site, features, target)
        X_dev, y_dev = split_and_scale(dev_site, features, target)

        for value in tune_values:
            start_time = time.time()
            temp_params = TRAINING_PARAMS.copy()
            temp_params[tune_param] = value

            lookback = temp_params["lookback"]
            X_train_seq = reshape_for_sequence(X_train, lookback)
            y_train_seq = y_train[lookback:]
            X_dev_seq = reshape_for_sequence(X_dev, lookback)
            y_dev_seq = y_dev[lookback:]

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_bilstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), 2, 64)
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
    results_df.to_csv(os.path.join(RESULTS_DIR, f"BiLSTM_HPT_{tune_param}_batch_{start_seq}_to_{start_seq + batch_size - 1}.csv"), index=False)

if __name__ == "__main__":
    train_df = pd.read_csv(DATA_PATH_TRAIN)
    dev_df = pd.read_csv(DATA_PATH_DEV)

    start_seq = 1
    batch_size = 5
    gauge_seq_ids = train_df["gauge_seq_id"].unique()
    batch_ids = gauge_seq_ids[start_seq - 1:start_seq - 1 + batch_size]

    run_architectural_tuning(train_df, dev_df, batch_ids)
    for param in TRAINING_PARAMS:
        run_training_param_tuning(train_df, dev_df, batch_ids, param, config["training_space"][param])
