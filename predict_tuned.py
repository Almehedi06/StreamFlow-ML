import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_loader import split_and_scale, reshape_for_sequence
import yaml

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type (LSTM, GRU, bilstm, conv1d)")
    parser.add_argument("--data", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--param_file", type=str, required=True, help="Path to tuned param CSV")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with saved models")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV for predictions")
    args = parser.parse_args()

    config = load_config()
    features = config["features"]
    target = config["target"]

    test_df = pd.read_csv(args.data)
    param_df = pd.read_csv(args.param_file)

    all_preds = []

    for _, row in param_df.iterrows():
        gid = row["gauge_seq_id"]
        print(f"🔍 Predicting for gauge_seq_id: {gid}")
        test_data = test_df[test_df["gauge_seq_id"] == gid]
        if test_data.empty:
            print(f"⚠️ No test data for gauge {gid}")
            continue

        # Extract tuned parameters
        lookback = int(row["lookback"])
        batch_size = int(row["batch_size"])
        row_dict = row.to_dict()

        # Preprocess test data
        X_scaled, y = split_and_scale(test_data, features, target)
        X_seq = reshape_for_sequence(X_scaled, lookback)
        y_seq = y[lookback:lookback + len(X_seq)]

        if len(X_seq) == 0:
            print(f"⚠️ Not enough test sequence for gauge {gid}")
            continue

        model_path = os.path.join(args.model_dir, f"{args.model}_{gid}.h5")
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            continue

        # Load model without compiling
        model = tf.keras.models.load_model(model_path, compile=False)

        preds = model.predict(X_seq, batch_size=batch_size).flatten()
        df_pred = pd.DataFrame({
            "gauge_seq_id": gid,
            "obs": y_seq,
            "pred": preds
        })

        all_preds.append(df_pred)

    if all_preds:
        final_df = pd.concat(all_preds, ignore_index=True)
        final_df.to_csv(args.output_csv, index=False)
        print(f"✅ Saved predictions to: {args.output_csv}")
    else:
        print("❌ No predictions were made.")
