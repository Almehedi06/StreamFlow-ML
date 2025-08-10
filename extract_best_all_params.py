import os
import glob
import pandas as pd
import argparse

def extract_best_all_params(model_name, results_dir='results', output_dir='results'):
    pattern = os.path.join(results_dir, f'{model_name}_HPT_*_batch_*.csv')
    all_files = glob.glob(pattern)

    param_data = {}

    for file_path in all_files:
        filename = os.path.basename(file_path)

        # Identify if it's an architectural param or tuning param
        if "_HPT_results_" in filename:
            # Architectural param (e.g., conv1d_HPT_results_batch_...)
            continue  # We'll handle these below in a separate loop
        else:
            param_name = filename.replace(f"{model_name}_HPT_", "").split("_batch_")[0]

            try:
                df = pd.read_csv(file_path)
                if 'gauge_seq_id' not in df.columns or 'dev_nse' not in df.columns:
                    print(f"⚠️ Skipping {filename}: missing required columns.")
                    continue

                for gauge_id, group in df.groupby('gauge_seq_id'):
                    best_row = group.loc[group['dev_nse'].idxmax()]
                    if gauge_id not in param_data:
                        param_data[gauge_id] = {}
                    param_data[gauge_id][param_name] = best_row[param_name]

            except Exception as e:
                print(f"❌ Error in {filename}: {e}")

    # Now handle architectural results separately
    arch_pattern = os.path.join(results_dir, f'{model_name}_HPT_results_batch_*.csv')
    arch_files = glob.glob(arch_pattern)

    for arch_path in arch_files:
        try:
            df = pd.read_csv(arch_path)
            if 'gauge_seq_id' not in df.columns or 'dev_nse' not in df.columns:
                print(f"⚠️ Skipping {arch_path}: missing required columns.")
                continue

            for gauge_id, group in df.groupby('gauge_seq_id'):
                best_row = group.loc[group['dev_nse'].idxmax()]
                if gauge_id not in param_data:
                    param_data[gauge_id] = {}

                # Determine arch params by model
                if model_name.lower() in ['lstm', 'gru', 'bilstm']:
                    for p in ['num_layers', 'neurons']:
                        if p in best_row:
                            param_data[gauge_id][p] = best_row[p]
                elif model_name.lower() == 'conv1d':
                    for p in ['num_layers', 'filters', 'kernel_size']:
                        if p in best_row:
                            param_data[gauge_id][p] = best_row[p]

        except Exception as e:
            print(f"❌ Error in {arch_path}: {e}")

    # Build final DataFrame
    if param_data:
        summary_df = pd.DataFrame.from_dict(param_data, orient='index')
        summary_df.index.name = 'gauge_seq_id'
        summary_df = summary_df.reset_index().sort_values(by='gauge_seq_id')

        # Define final column order
        base_cols = ['gauge_seq_id']
        tuning_cols = ['activation', 'loss_function', 'dropout', 'batch_size', 'epochs',
                       'lookback', 'learning_rate', 'optimizer']
        if model_name.lower() in ['lstm', 'gru', 'bilstm']:
            arch_cols = ['num_layers', 'neurons']
        elif model_name.lower() == 'conv1d':
            arch_cols = ['num_layers', 'filters', 'kernel_size']
        else:
            arch_cols = []

        ordered_cols = base_cols + tuning_cols + arch_cols
        summary_df = summary_df[[col for col in ordered_cols if col in summary_df.columns]]

        output_file = os.path.join(output_dir, f'best_{model_name}_all_params_by_gauge.csv')
        summary_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")
    else:
        print("❌ No valid parameter results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name (e.g., LSTM, GRU, bilstm, conv1d)')
    parser.add_argument('--results_dir', default='results', help='Directory where result CSVs are stored')
    args = parser.parse_args()

    extract_best_all_params(model_name=args.model, results_dir=args.results_dir)
