import json
import subprocess
import os
import pandas as pd

# Percorsi
config_path = "src/ML_analysis/config/ml_config.json"
script_path = "src/ML_analysis/analysis/classification.py"
results_base = "results/ML"

# Combinazioni da eseguire
# ORDINE: Networks prima, poi Voxel+UMAP
configs = [
    # === NETWORKS (no UMAP) ===
    {"dataset_type": "networks", "umap": False, "group1": "AD", "group2": "PSP"},
    {"dataset_type": "networks", "umap": False, "group1": "AD", "group2": "CBS"},
    {"dataset_type": "networks", "umap": False, "group1": "PSP", "group2": "CBS"},
    
    # === VOXEL + UMAP ===
    {"dataset_type": "voxel", "umap": True, "group1": "AD", "group2": "PSP"},
    {"dataset_type": "voxel", "umap": True, "group1": "AD", "group2": "CBS"},
    {"dataset_type": "voxel", "umap": True, "group1": "PSP", "group2": "CBS"},
]

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
    print(f"\n>>> Running classification.py for {cfg['group1']} vs {cfg['group2']} | {cfg['dataset_type'].upper()}")
    p = subprocess.run(["python", script_path], env={**os.environ, "PYTHONPATH": "src"})


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
