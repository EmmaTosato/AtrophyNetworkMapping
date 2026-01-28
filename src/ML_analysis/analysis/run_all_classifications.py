import json
import subprocess
import os

# Percorsi
config_path = "src/ML_analysis/config/ml_config.json"
script_path = "src/ML_analysis/analysis/classification.py"

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

