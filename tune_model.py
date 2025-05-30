import argparse
import importlib
import pandas as pd
import yaml
import os

def load_config():
    with open("hpt/config.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Generic HPT runner for DL models")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., lstm, bilstm, gru, conv1d)")
    parser.add_argument("--start_seq", type=int, required=True, help="Start gauge_seq_id")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--hpt_type", type=str, choices=["arch", "train", "all"], default="all",
                        help="HPT type: 'arch' for architectural, 'train' for training parameters, 'all' for both")
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

    if args.hpt_type in ["arch", "all"]:
        print(f"Running architectural HPT for {model_name} on gauge_seq_id {batch_ids}")
        module.run_architectural_tuning(train_df, dev_df, batch_ids)

    if args.hpt_type in ["train", "all"]:
        print(f"Running training parameter sensitivity analysis for {model_name} on gauge_seq_id {batch_ids}")
        for param in config["training_space"]:
            print(f"Tuning {param} for {model_name}...")
            module.run_training_param_tuning(train_df, dev_df, batch_ids, param, config["training_space"][param])

if __name__ == "__main__":
    main()
