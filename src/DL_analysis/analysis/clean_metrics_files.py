import os
import json
import glob

def clean_metrics_files(base_dir):
    # Find all metrics.json files recursively
    # Structure: results/DL/{pair}/{model}/fold_{i}/metrics.json
    search_pattern = os.path.join(base_dir, "*", "*", "*", "metrics.json")
    files = glob.glob(search_pattern)

    print(f"Found {len(files)} metrics.json files.")

    for json_file in files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'metrics' in data:
                print(f"Cleaning {json_file}...")
                del data['metrics']
                
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                print(f"Skipping {json_file} (no metrics key found).")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    base_output_dir = "results/DL"
    if os.path.exists(base_output_dir):
        clean_metrics_files(base_output_dir)
    else:
        print(f"Directory {base_output_dir} not found.")
