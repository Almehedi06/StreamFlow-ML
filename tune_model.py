# tune_model.py

import argparse
import importlib
import pandas as pd
import yaml
import os

# Load config
def load_config():
    with open("hpt/config.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Generic HPT runner for DL models")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., lstm, bilstm, gru, conv1d)")
    parser.add_argument("--start_seq", type=int, required=True, help="Start gauge_seq_id")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    args = parser.parse_args()

    config = load_config()
    model_name = args.model.lower()

    try:
        module = importlib.import_module(f"hpt.{model_name}_tune")
    except ModuleNotFoundError:
        raise Exception(f"No module found for model: {model_name}. Please ensure hpt/{model_name}_tune.py exists.")

    train_df = pd.read_csv(config["paths"]["train"])
    dev_df = pd.read_csv(config["paths"]["dev"])
    gauge_seq_ids = train_df["gauge_seq_id"].unique()
    batch_ids = gauge_seq_ids[args.start_seq - 1: args.start_seq - 1 + args.batch_size]

    print(f"Running architectural HPT for {model_name} on gauge_seq_id {batch_ids}")
    module.run_architectural_tuning(train_df, dev_df, batch_ids)

    if model_name in config["architectural_params"]:
        for param in config["training_params"]:
            print(f"Tuning {param} for {model_name}...")
            module.run_training_param_tuning(train_df, dev_df, batch_ids, param, config["training_space"][param])
    else:
        print(f"Skipping training param tuning for {model_name} (not supported in config).")

if __name__ == "__main__":
    main()
