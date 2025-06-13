# tune_model.py

import argparse
import os
import pandas as pd
from hpt.gru_tunehp import run_full_tuning, save_best_hyperparameters
import yaml

# Load config
with open("hpt/config.yaml", "r") as file:
    config = yaml.safe_load(file)

TRAIN_PATH = config["paths"]["train"]
DEV_PATH = config["paths"]["dev"]
RESULTS_DIR = config["paths"]["results"]
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Run HPT for GRU")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gru)")
    parser.add_argument("--start_seq", type=int, required=True, help="Starting index for site batch")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of sites in batch")
    parser.add_argument("--hpt_type", type=str, default="all", choices=["all"], help="Tuning type (currently only supports 'all')")
    args = parser.parse_args()

    if args.model != "gru":
        raise ValueError("Only GRU model is supported in this version.")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)

    all_site_ids = sorted(train_df["gauge_seq_id"].unique())
    batch_ids = all_site_ids[args.start_seq: args.start_seq + args.batch_size]

    print(f"🔄 Running GRU HPT for gauge_seq_id: {batch_ids[0]} to {batch_ids[-1]}")
    results_df = run_full_tuning(train_df, dev_df, batch_ids)

    output_path = os.path.join(RESULTS_DIR, f"gru_besthp_{batch_ids[0]}_to_{batch_ids[-1]}.csv")
    save_best_hyperparameters(results_df, output_path)

if __name__ == "__main__":
    main()
