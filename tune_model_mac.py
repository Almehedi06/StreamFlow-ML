import argparse
import os
import time
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils.evaluation import calculate_nse

# Set global seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load config
with open("hpt/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRAINING_PARAMS = config["training_params"]
ARCHITECTURAL_PARAMS = config["architectural_params"]["lstm"]
DATA_PATH_TRAIN = config["paths"]["train"]
DATA_PATH_DEV = config["paths"]["dev"]
RESULTS_DIR = config["paths"]["results"]
features = config["features"]
target = config["target"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# Build LSTM
def build_lstm_model(input_shape, num_layers, neurons, training_params):
    model = Sequential()
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        if i == 0:
            model.add(LSTM(neurons, activation=training_params["activation"], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(neurons, activation=training_params["activation"], return_sequences=return_seq))
        model.add(Dropout(training_params["dropout"]))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=training_params["learning_rate"])
    model.compile(optimizer=optimizer, loss=training_params["loss_function"])
    return model

# Create sequences
def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Architectural tuning
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

        for num_layers, neurons in product(ARCHITECTURAL_PARAMS["num_layers"], ARCHITECTURAL_PARAMS["neurons"]):
            start_time = time.time()
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, TRAINING_PARAMS["lookback"])
            X_dev_seq, y_dev_seq = create_sequences(X_dev, y_dev, TRAINING_PARAMS["lookback"])

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), num_layers, neurons, TRAINING_PARAMS)
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

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, f"LSTM_HPT_arch_batch_{batch_ids[0]}_to_{batch_ids[-1]}.csv"), index=False)

# Training parameter tuning
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
            temp_params = TRAINING_PARAMS.copy()
            temp_params[tune_param] = value

            lookback = temp_params["lookback"]
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
            X_dev_seq, y_dev_seq = create_sequences(X_dev, y_dev, lookback)

            if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                continue

            model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), 2, 64, temp_params)
            model.fit(X_train_seq, y_train_seq, epochs=temp_params["epochs"], batch_size=temp_params["batch_size"], validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)

            train_nse = calculate_nse(y_train_seq, model.predict(X_train_seq).flatten())
            dev_nse = calculate_nse(y_dev_seq, model.predict(X_dev_seq).flatten())

            results.append({
                "gauge_seq_id": gid,
                tune_param: value,
                "train_nse": train_nse,
                "dev_nse": dev_nse,
                "nse_difference": train_nse - dev_nse,
                "time_sec": round(time.time() - time.time(), 2)
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, f"LSTM_HPT_{tune_param}_batch_{batch_ids[0]}_to_{batch_ids[-1]}.csv"), index=False)

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm"])
    parser.add_argument("--hpt_type", type=str, required=True, choices=["arch", "train", "all"])
    parser.add_argument("--tune_param", type=str, help="(only for --hpt_type train)")
    parser.add_argument("--start_seq", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(DATA_PATH_TRAIN)
    dev_df = pd.read_csv(DATA_PATH_DEV)
    batch_ids = list(range(args.start_seq, args.start_seq + args.batch_size))

    if args.hpt_type == "arch":
        run_architectural_tuning(train_df, dev_df, batch_ids)

    elif args.hpt_type == "train":
        if not args.tune_param:
            raise ValueError("Please provide --tune_param when hpt_type is train")
        tune_values = config["training_space"][args.tune_param]
        run_training_param_tuning(train_df, dev_df, batch_ids, args.tune_param, tune_values)

    elif args.hpt_type == "all":
        for tune_param, tune_values in config["training_space"].items():
            print(f"Tuning {tune_param} ...")
            run_training_param_tuning(train_df, dev_df, batch_ids, tune_param, tune_values)

if __name__ == "__main__":
    main()
