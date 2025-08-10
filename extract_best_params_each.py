import os
import glob
import pandas as pd
import argparse

def extract_best_tuned_params(model_name, results_dir='results', output_dir='results'):
    # Search pattern for tuning CSVs
    pattern = os.path.join(results_dir, f'{model_name}_HPT_*_batch_*.csv')
    tuning_files = glob.glob(pattern)

    param_data = {}

    for file_path in tuning_files:
        filename = os.path.basename(file_path)

        # Skip architectural results files
        if "_HPT_results_" in filename:
            continue

        # Get tuned parameter name
        param_name = filename.replace(f"{model_name}_HPT_", "").split("_batch_")[0]

        try:
            df = pd.read_csv(file_path)
            if 'gauge_seq_id' not in df.columns or 'dev_nse' not in df.columns:
                print(f" Skipping {filename}: missing required columns.")
                continue

            for gauge_id, group in df.groupby('gauge_seq_id'):
                best_row = group.loc[group['dev_nse'].idxmax()]
                if gauge_id not in param_data:
                    param_data[gauge_id] = {}

                # Store best value of this parameter
                param_data[gauge_id][param_name] = best_row[param_name]

                # Store overall best dev_nse and corresponding metrics
                current_best = param_data[gauge_id].get('dev_nse', -999)
                if best_row['dev_nse'] > current_best:
                    param_data[gauge_id]['train_nse'] = best_row['train_nse']
                    param_data[gauge_id]['dev_nse'] = best_row['dev_nse']
                    param_data[gauge_id]['nse_difference'] = best_row['nse_difference']
                    param_data[gauge_id]['time_sec'] = best_row['time_sec']

        except Exception as e:
            print(f" Error in {filename}: {e}")

    # Save results
    if param_data:
        summary_df = pd.DataFrame.from_dict(param_data, orient='index')
        summary_df.index.name = 'gauge_seq_id'
        summary_df = summary_df.reset_index().sort_values(by='gauge_seq_id')

        # Enforce consistent column order
        preferred_columns = [
            'gauge_seq_id', 'activation', 'train_nse', 'dev_nse', 'nse_difference', 'time_sec',
            'loss_function', 'dropout', 'batch_size', 'epochs', 'lookback', 'learning_rate', 'optimizer'
        ]
        summary_df = summary_df[[col for col in preferred_columns if col in summary_df.columns]]

        output_file = os.path.join(output_dir, f'best_{model_name}_tuned_params_by_gauge.csv')
        summary_df.to_csv(output_file, index=False)
        print(f" Saved: {output_file}")
    else:
        print(" No valid tuning results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name (e.g., CONV1D, LSTM, GRU, BILSTM)')
    parser.add_argument('--results_dir', default='results', help='Directory where tuning CSVs are stored')
    args = parser.parse_args()

    extract_best_tuned_params(model_name=args.model, results_dir=args.results_dir)
