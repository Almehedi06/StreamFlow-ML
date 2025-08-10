import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import split_and_scale, reshape_for_sequence
from utils.evaluation import calculate_metrics
import yaml
import os

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., lstm, gru, bilstm, conv1d)")
    parser.add_argument("--data", type=str, default="data/camel_dev.csv")
    parser.add_argument("--start_seq", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    features = config["features"]
    target = config["target"]

    df = pd.read_csv(args.data)
    all_ids = df["gauge_seq_id"].unique()
    batch_ids = all_ids[args.start_seq - 1: args.start_seq - 1 + args.batch_size]

    timesteps = config[args.model]["lookback"]
    result_metrics = []
    result_series = []

    model_dir = f"saved_models/base_{args.model.upper()}"
    os.makedirs("results", exist_ok=True)

    for gid in batch_ids:
        print(f" Predicting for gauge_seq_id {gid}")
        test_data = df[df["gauge_seq_id"] == gid].copy()
        if test_data.empty:
            continue

        X_scaled, y_true = split_and_scale(test_data, features, target)
        X_seq = reshape_for_sequence(X_scaled, timesteps)
        y_true_seq = y_true[timesteps:timesteps + len(X_seq)]

        # Updated model filename format
        model_path = os.path.join(model_dir, f"{args.model}_{gid}.h5")
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            continue

        model = load_model(model_path, compile=False)
        y_pred = model.predict(X_seq).flatten()

        # Evaluation metrics
        metrics = {"gauge_seq_id": gid, **calculate_metrics(y_true_seq, y_pred)}
        result_metrics.append(metrics)

        # Full predicted vs true time series
        dates = test_data.iloc[timesteps:timesteps + len(X_seq)].get("date", pd.Series([np.nan]*len(y_pred)))
        for i in range(len(y_pred)):
            result_series.append({
                "gauge_seq_id": gid,
                "date": dates.iloc[i] if not pd.isna(dates.iloc[i]) else i,
                "y_true": y_true_seq[i],
                "y_pred": y_pred[i]
            })

    # Save both outputs
    df_metrics = pd.DataFrame(result_metrics)
    df_series = pd.DataFrame(result_series)

    base_path = f"results/prediction_metrics_{args.model}_batch_{args.start_seq}_to_{args.start_seq + args.batch_size - 1}"
    df_metrics.to_csv(base_path + ".csv", index=False)
    df_series.to_csv(base_path + "_series.csv", index=False)

    print(f" Metrics saved to: {base_path}.csv")
    print(f" Time series saved to: {base_path}_series.csv")
