import json
import subprocess
import os
import pandas as pd

import argparse

# Percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))

# Default configs
default_config = os.path.join(project_root, "src/ML_analysis/config/ml_config.json")
default_run_config = os.path.join(project_root, "src/ML_analysis/config/ml_run_all_config.json")
script_path = os.path.join(script_dir, "classification.py")

# Argparse
parser = argparse.ArgumentParser(description="Run all classifications batch script")
parser.add_argument("--config", default=default_config, help="Main ML Config Path")
parser.add_argument("--run_config", default=default_run_config, help="Running combinations config (ml_run_all_config.json)")
args = parser.parse_args()

config_path = args.config
run_config_path = args.run_config

# Load base output dir from config
with open(config_path, "r") as f:
    _temp_cfg = json.load(f)
results_base = _temp_cfg.get("fixed_parameters", {}).get("output_dir", "results/ML/")

# Load "Job" configuration
run_config_path = os.path.join(project_root, "src/ML_analysis/config/ml_run_all_config.json")
if os.path.exists(run_config_path):
    with open(run_config_path, "r") as f:
        run_cfg = json.load(f)
    print(f"Loaded run config from: {run_config_path}")
    configs = run_cfg.get("classification", {}).get("runs", [])
else:
    print(f"Warning: Run config not found at {run_config_path}. Using empty list.")
    configs = []

# Loop su ciascuna configurazione
for cfg in configs:
    # Carica config.json
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Modifica la sezione "job"
    full_config["job"]["dataset_type"] = cfg["dataset_type"]
    full_config["job"]["task_type"] = "classification"
    full_config["job"]["threshold"] = False
    full_config["job"]["umap"] = cfg["umap"]

    # Modifica la sezione "classification"
    full_config["classification"] = {
        "tuning": True,
        "permutation_test": True,
        "n_permutations": 1000,
        "perm_cv": 5,
        "group1": cfg["group1"],
        "group2": cfg["group2"],
        "n_folds": 5,
        "RandomForest": None,
        "GradientBoosting": None,
        "KNN": None
    }

    # Salva nuova configurazione
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    # Esegui lo script classification.py
    p = subprocess.run(
        ["python", script_path], 
        env={**os.environ, "PYTHONPATH": os.path.join(project_root, "src")}
    )


# === AGGREGATE RESULTS PER DATASET TYPE ===
print("\n" + "=" * 60)
print("AGGREGATING RESULTS...")
print("=" * 60)

for dataset_type in ["networks", "voxel"]:
    if dataset_type == "networks":
        method_folder = "classification"
    else:
        method_folder = "umap_classification"
    
    base_path = os.path.join(results_base, dataset_type, method_folder)
    all_summaries = []
    
    # Find all summary_results.csv
    for pair_folder in os.listdir(base_path):
        pair_path = os.path.join(base_path, pair_folder)
        summary_path = os.path.join(pair_path, "summary_results.csv")
        
        if os.path.isdir(pair_path) and os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df["comparison"] = pair_folder
            all_summaries.append(df)
    
    if all_summaries:
        df_total = pd.concat(all_summaries, ignore_index=True)
        output_path = os.path.join(base_path, "total_aggregated_results.csv")
        df_total.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

print("\n>>> BATCH COMPLETED!")
