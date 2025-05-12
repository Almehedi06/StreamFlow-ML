import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import split_and_scale, reshape_for_sequence
from utils.evaluation import calculate_metrics
import yaml

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., lstm)")
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
    result_rows = []

    for gid in batch_ids:
        print(f"🔍 Predicting for gauge_seq_id {gid}")
        test_data = df[df["gauge_seq_id"] == gid]
        if test_data.empty:
            continue

        X_scaled, y_true = split_and_scale(test_data, features, target)
        X_seq = reshape_for_sequence(X_scaled, timesteps)
        y_true_seq = y_true[timesteps:timesteps + len(X_seq)]

        model_path = f"saved_models/{args.model}_{gid}.h5"
        model = load_model(model_path, compile=False)
        y_pred = model.predict(X_seq).flatten()

        metrics = {"gauge_seq_id": gid, **calculate_metrics(y_true_seq, y_pred)}
        result_rows.append(metrics)

    df_results = pd.DataFrame(result_rows)
    df_results.to_csv(f"results/prediction_metrics_{args.model}_batch_{args.start_seq}_to_{args.start_seq + args.batch_size - 1}.csv", index=False)
    print("✅ Predictions saved.")
