import os
import json
import pandas as pd
import glob

def process_results(base_dir):
    # Find all aggregated_results.json files recursively
    # Structure: results/DL/{pair}/{model}/aggregated_results.json
    search_pattern = os.path.join(base_dir, "*", "*", "aggregated_results.json")
    files = glob.glob(search_pattern)

    print(f"Found {len(files)} result files.")

    for json_file in files:
        folder_path = os.path.dirname(json_file)
        print(f"Processing: {folder_path}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 1. aggregated_results.csv
            if 'aggregate' in data:
                agg_data = data['aggregate']
                # Round values to 3 decimals
                agg_data_rounded = {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in agg_data.items()}
                
                df_agg = pd.DataFrame([agg_data_rounded])
                csv_agg_path = os.path.join(folder_path, "aggregated_results.csv")
                df_agg.to_csv(csv_agg_path, index=False)
                print(f"  -> Created aggregated_results.csv")

            # 2. nested_cv_results.csv & 3. best_params.csv
            if 'folds' in data:
                folds_data = data['folds']
                
                # Lists to hold rows
                metric_rows = []
                param_rows = []

                for fold in folds_data:
                    # Metrics
                    metric_row = {'fold': fold['fold']}
                    if 'metrics' in fold:
                        for k, v in fold['metrics'].items():
                            metric_row[k] = round(v, 3) if isinstance(v, (int, float)) else v
                    metric_rows.append(metric_row)

                    # Params
                    param_row = {'fold': fold['fold']}
                    if 'best_params' in fold:
                        for k, v in fold['best_params'].items():
                             param_row[k] = v 
                    param_rows.append(param_row)
                
                # Save Nested CV Metrics
                df_folds = pd.DataFrame(metric_rows)
                csv_folds_path = os.path.join(folder_path, "nested_cv_results.csv")
                df_folds.to_csv(csv_folds_path, index=False)
                print(f"  -> Created nested_cv_results.csv")



        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    base_output_dir = "results/DL"
    if os.path.exists(base_output_dir):
        process_results(base_output_dir)
    else:
        print(f"Directory {base_output_dir} not found.")
