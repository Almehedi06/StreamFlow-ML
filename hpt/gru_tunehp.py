import os
import time
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils.data_loader import reshape_for_sequence
from utils.evaluation import calculate_nse

# --- Set seed for reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Load config ---
with open("hpt/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRAINING_PARAMS = config["training_params"]              # Defaults
TRAINING_SPACE = config["training_space"]                # Tuning space
ARCHITECTURAL_PARAMS = config["architectural_params"]["gru"]

DATA_PATH_TRAIN = config["paths"]["train"]
DATA_PATH_DEV = config["paths"]["dev"]
RESULTS_DIR = config["paths"]["results"]
os.makedirs(RESULTS_DIR, exist_ok=True)

features = config["features"]
target = config["target"]

def build_gru_model(input_shape, num_layers, neurons, activation, dropout, optimizer_name, learning_rate, loss_function):
    model = Sequential()
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        if i == 0:
            model.add(GRU(neurons, activation=activation, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(GRU(neurons, activation=activation, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "adam" else RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

def run_full_tuning(train_df, dev_df, batch_ids):
    all_results = []

    for gid in batch_ids:
        train_site = train_df[train_df["gauge_seq_id"] == gid].copy()
        dev_site = dev_df[dev_df["gauge_seq_id"] == gid].copy()
        if train_site.empty or dev_site.empty:
            continue

        X_train = StandardScaler().fit_transform(train_site[features])
        y_train = train_site[target].values.ravel()
        X_dev = StandardScaler().fit_transform(dev_site[features])
        y_dev = dev_site[target].values.ravel()

        for num_layers, neurons, activation, dropout, batch_size, epochs, learning_rate, loss_fn, lookback, optimizer_name in product(
            ARCHITECTURAL_PARAMS["num_layers"],
            ARCHITECTURAL_PARAMS["neurons"],
            TRAINING_SPACE["activation"],
            TRAINING_SPACE["dropout"],
            TRAINING_SPACE["batch_size"],
            TRAINING_SPACE["epochs"],
            TRAINING_SPACE["learning_rate"],
            TRAINING_SPACE["loss_function"],
            TRAINING_SPACE["lookback"],
            TRAINING_SPACE["optimizer"]
        ):
            try:
                X_train_seq = reshape_for_sequence(X_train, lookback)
                X_dev_seq = reshape_for_sequence(X_dev, lookback)
                y_train_seq = y_train[lookback:]
                y_dev_seq = y_dev[lookback:]

                if len(X_train_seq) == 0 or len(X_dev_seq) == 0:
                    continue

                model = build_gru_model(
                    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                    num_layers=num_layers,
                    neurons=neurons,
                    activation=activation,
                    dropout=dropout,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    loss_function=loss_fn
                )

                model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
                          validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)

                train_nse = calculate_nse(y_train_seq, model.predict(X_train_seq).flatten())
                dev_nse = calculate_nse(y_dev_seq, model.predict(X_dev_seq).flatten())

                all_results.append({
                    "gauge_seq_id": gid,
                    "num_layers": num_layers,
                    "neurons": neurons,
                    "activation": activation,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "loss_function": loss_fn,
                    "lookback": lookback,
                    "optimizer": optimizer_name,
                    "train_nse": train_nse,
                    "dev_nse": dev_nse,
                    "nse_difference": train_nse - dev_nse
                })
            except Exception as e:
                print(f"⚠️ Error in training for gauge {gid}: {e}")

    return pd.DataFrame(all_results)

def save_best_hyperparameters(df_all, output_path):
    best_df = df_all.sort_values("dev_nse", ascending=False).drop_duplicates("gauge_seq_id")
    drop_cols = ["train_nse", "dev_nse", "nse_difference"]
    best_df = best_df.drop(columns=drop_cols)
    best_df = best_df.sort_values("gauge_seq_id").reset_index(drop=True)
    best_df.to_csv(output_path, index=False)
    print(f"✅ Saved best hyperparameters to: {output_path}")

def main():
    train_df = pd.read_csv(DATA_PATH_TRAIN)
    dev_df = pd.read_csv(DATA_PATH_DEV)
    
    all_site_ids = sorted(train_df["gauge_seq_id"].unique())
    batch_ids = all_site_ids[100:120]  # change if needed

    all_results = run_full_tuning(train_df, dev_df, batch_ids)
    if all_results.empty:
        print("❌ No results produced.")
        return

    output_path = os.path.join(RESULTS_DIR, "gru_besthp.csv")
    save_best_hyperparameters(all_results, output_path)

if __name__ == "__main__":
    main()
