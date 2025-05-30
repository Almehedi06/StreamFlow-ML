import argparse
import os
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

from utils.data_loader import split_and_scale, reshape_for_sequence
from utils.evaluation import calculate_metrics

from models.lstm_model import build_lstm_model
from models.gru_model import build_gru_model
from models.bilstm_model import build_bilstm_model
# from models.conv1d_model import build_conv1d_model  # uncomment when added

model_map = {
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "bilstm": build_bilstm_model,
    # "conv1d": build_conv1d_model
}

def train_model(df, model_name, model_config, features, target, gauge_ids):
    builder = model_map.get(model_name)
    if builder is None:
        raise ValueError(f"Model {model_name} not implemented.")

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    timesteps = model_config.get("lookback", 20)
    all_metrics = []

    for gauge_id in gauge_ids:
        print(f"\n Training {model_name.upper()} for Gauge Seq ID: {gauge_id}")
        site_data = df[df['gauge_seq_id'] == gauge_id]
        split_index = int(len(site_data) * 0.8)

        train_data = site_data.iloc[:split_index]
        test_data = site_data.iloc[split_index:]

        X_train, y_train = split_and_scale(train_data, features, target)
        X_test, y_test = split_and_scale(test_data, features, target)

        X_train = reshape_for_sequence(X_train, timesteps)
        X_test = reshape_for_sequence(X_test, timesteps)
        y_train = y_train[timesteps:timesteps + len(X_train)]
        y_test = y_test[timesteps:timesteps + len(X_test)]

        model = builder(input_shape=(timesteps, X_train.shape[2]), config=model_config)

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train,
                  validation_split=0.1,
                  epochs=model_config.get("epochs", 50),
                  batch_size=model_config.get("batch_size", 128),
                  callbacks=[early_stop],
                  verbose=1)

        y_pred = model.predict(X_test).flatten()
        metrics = {"gauge_seq_id": gauge_id, **calculate_metrics(y_test, y_pred)}
        all_metrics.append(metrics)

        # Save model
        model_filename = f"saved_models/{model_name}_{gauge_id}.h5"
        model.save(model_filename)
        print(f" Saved model: {model_filename}")

        print(f" Done Gauge {gauge_id}: R²={metrics['r2']:.3f}, NSE={metrics['nse']:.3f}")

    return pd.DataFrame(all_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train: lstm | gru | bilstm")
    parser.add_argument("--data", type=str, default="data/camel_train.csv")
    parser.add_argument("--start_seq", type=int, default=1, help="Starting gauge_seq_id")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of gauge stations to use")
    args = parser.parse_args()

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config[args.model]
    df = pd.read_csv(args.data)

    features = config["features"]
    target = config["target"]
    all_seq_ids = df["gauge_seq_id"].unique()
    batch_ids = all_seq_ids[args.start_seq - 1: args.start_seq - 1 + args.batch_size]

    results = train_model(df, args.model, model_config, features, target, batch_ids)
    output_csv = f"results/metrics_{args.model}_seq_{args.start_seq}_to_{args.start_seq + args.batch_size - 1}.csv"
    results.to_csv(output_csv, index=False)
    print(f" Metrics saved to: {output_csv}")
