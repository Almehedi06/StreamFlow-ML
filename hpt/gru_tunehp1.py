import os
import pandas as pd
from glob import glob

# Folder where HPT CSVs are stored
HPT_DIR = "results"

# Hyperparameter file patterns
param_files = {
    "activation": "GRU_HPT_activation_batch_*.csv",
    "batch_size": "GRU_HPT_batch_size_batch_*.csv",
    "dropout": "GRU_HPT_dropout_batch_*.csv",
    "epochs": "GRU_HPT_epochs_batch_*.csv",
    "learning_rate": "GRU_HPT_learning_rate_batch_*.csv",
    "loss_function": "GRU_HPT_loss_function_batch_*.csv",
    "lookback": "GRU_HPT_lookback_batch_*.csv",
    "optimizer": "GRU_HPT_optimizer_batch_*.csv",
    "results": "GRU_HPT_results_batch_*.csv"  # architectural: num_layers + neurons
}

def load_best_from_file(path, cols):
    df = pd.read_csv(path)
    if "dev_nse" not in df.columns:
        print(f"⚠️ dev_nse column missing in {path}")
        return pd.DataFrame()
    best_df = df.sort_values("dev_nse", ascending=False).drop_duplicates("gauge_seq_id")
    return best_df[["gauge_seq_id"] + cols]

def main():
    best_param_dfs = []

    for param, pattern in param_files.items():
        files = sorted(glob(os.path.join(HPT_DIR, pattern)))
        if not files:
            print(f"⚠️ No files found for {param}")
            continue

        dfs = []
        for file in files:
            if param == "results":
                cols = ["num_layers", "neurons"]
            else:
                cols = [param]
            df = load_best_from_file(file, cols)
            dfs.append(df)

        combined = pd.concat(dfs)
        combined = combined.drop_duplicates("gauge_seq_id", keep="first")
        best_param_dfs.append(combined)

    # Merge all on gauge_seq_id
    final_df = best_param_dfs[0]
    for df in best_param_dfs[1:]:
        final_df = pd.merge(final_df, df, on="gauge_seq_id", how="outer")

    final_df = final_df.sort_values("gauge_seq_id").reset_index(drop=True)

    # Save final output
    output_path = os.path.join(HPT_DIR, "gru_best_hyperparameters.csv")
    final_df.to_csv(output_path, index=False)
    print(f"✅ Final best hyperparameters saved at: {output_path}")

if __name__ == "__main__":
    main()
