import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from utils.data_loader import split_and_scale, reshape_for_sequence
import yaml

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def build_model(model_type, input_shape, params):
    model = models.Sequential()
    num_layers = int(params.get("num_layers", 1))
    dropout = float(params.get("dropout", 0.2))
    activation = params.get("activation", "relu")
    optimizer = params.get("optimizer", "adam")
    loss_function = params.get("loss_function", "mse")

    if model_type.lower() in ["lstm", "gru", "bilstm"]: 
        RNNLayer = {
            "lstm": layers.LSTM,
            "gru": layers.GRU,
            "bilstm": lambda *args, **kwargs: layers.Bidirectional(layers.LSTM(*args, **kwargs))
        }[model_type.lower()]
        neurons = int(params.get("neurons", 64))
        for i in range(num_layers):
            return_seq = i < (num_layers - 1)
            if i == 0:
                model.add(RNNLayer(neurons, return_sequences=return_seq, input_shape=input_shape))
            else:
                model.add(RNNLayer(neurons, return_sequences=return_seq))
            model.add(layers.Dropout(dropout))

    elif model_type.lower() == "conv1d":
        filters = int(params.get("filters", 64))
        kernel_size = int(params.get("kernel_size", 3))
        for i in range(num_layers):
            if i == 0:
                model.add(layers.Conv1D(filters, kernel_size, activation=activation, input_shape=input_shape))
            else:
                model.add(layers.Conv1D(filters, kernel_size, activation=activation))
            model.add(layers.Dropout(dropout))
        model.add(layers.Flatten())

    model.add(layers.Dense(1))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["mae"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., LSTM, GRU, bilstm, conv1d)")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data (e.g., data/camel_train.csv)")
    parser.add_argument("--param_file", type=str, help="Path to param file (default: results/best_<model>_all_params_by_gauge.csv)")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="Base directory to save trained models")
    args = parser.parse_args()

    config = load_config()
    features = config["features"]
    target = config["target"]

    param_csv = args.param_file or f"results/best_{args.model}_all_params_by_gauge.csv"
    param_df = pd.read_csv(param_csv)
    df = pd.read_csv(args.data)

    model_name = args.model
    save_subdir = os.path.join(args.output_dir, f"tuned_{model_name}")
    os.makedirs(save_subdir, exist_ok=True)

    for _, row in param_df.iterrows():
        gid = row["gauge_seq_id"]
        print(f"\n🚀 Training model for gauge_seq_id: {gid}")
        train_data = df[df["gauge_seq_id"] == gid]
        if train_data.empty:
            print(f"⚠️ No data for gauge {gid}")
            continue

        input_params = row.to_dict()

        for key in ["num_layers", "neurons", "filters", "kernel_size", "epochs", "batch_size", "lookback"]:
            if key in input_params:
                input_params[key] = int(input_params[key])
        if "dropout" in input_params:
            input_params["dropout"] = float(input_params["dropout"])

        lookback = input_params.get("lookback", 10)
        epochs = input_params.get("epochs", 50)
        batch_size = input_params.get("batch_size", 32)

        X_scaled, y = split_and_scale(train_data, features, target)
        X_seq = reshape_for_sequence(X_scaled, lookback)
        y_seq = y[lookback:lookback + len(X_seq)]

        if len(X_seq) == 0:
            print(f"⚠️ Not enough data for gauge {gid} (lookback={lookback})")
            continue

        input_shape = (X_seq.shape[1], X_seq.shape[2])

        try:
            model = build_model(model_name, input_shape, input_params)
            model.fit(X_seq, y_seq, batch_size=batch_size, epochs=epochs, verbose=0)

            out_path = os.path.join(save_subdir, f"{model_name}_{gid}.h5")
            model.save(out_path)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Failed to train gauge {gid}: {e}")
            continue
