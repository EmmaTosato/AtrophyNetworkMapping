# Deep Learning Configuration Reference

This document details the configuration files used by the pipeline: `src/DL_analysis/config/cnn.json` (Execution) and `src/DL_analysis/config/cnn_grid.json` (Hyperparameters).

## 1. Experiments & Environment (`cnn.json`)

**Function:** Defines the list of experiments (Jobs) to run and file paths.
**Location:** `src/DL_analysis/config/cnn.json`

### Structure
```json
{
    "global": {
        "base_output_dir": "results/DL",
        "data_dir": "/data/users/etosato/ANM_Verona/data/FCmaps_processed",
        "metadata_path": "/data/users/etosato/ANM_Verona/assets/metadata/labels.csv"
    },
    "experiments": [
        {
            "group1": "AD",
            "group2": "PSP",
            "models": ["resnet", "alexnet"]
        },
        {
            "group1": "CN",
            "group2": "AD",
            "models": ["vgg16"]
        }
    ]
}
```

### Key Fields
*   `global.base_output_dir`: Root directory for all results.
*   `experiments`: List of job objects.
*   `group1`/`group2`: Diagnostic labels to compare (must match CSV metadata).
*   `models`: List of architectures to train for this comparison.

---

## 2. Hyperparameters (`cnn_grid.json`)

**Function:** Defines the grid search space for each model architecture.
**Location:** `src/DL_analysis/config/cnn_grid.json`

### Structure
Keys are model names (`resnet`, `alexnet`, `vgg16`). Values are dictionaries where **lists** denote the search space.

*   `[0.01]`: Fixed value (no tuning).
*   `[0.01, 0.001]`: Grid search will test both values.

```json
"resnet": {
    "lr": [0.01],                 // Learning Rate
    "batch_size": [8],            // Batch Size
    "weight_decay": [0.0001],     // L2 Regularization
    "epochs": [40],               // Training Epochs
    "optimizer": ["sgd"],         // Optimizer (sgd/adam)
    "momentum": [0.9],            // SGD Momentum
    "scheduler_gamma": [0.1],     // LR Scheduler Decay Factor
    "scheduler_patience": [5],    // LR Scheduler Patience
    "patience": [null]            // Early Stopping (null = disabled)
}
```

### Usage
*   To **fix** a parameter: Use a single-item list (e.g., `[0.01]`).
*   To **tune** a parameter: Use a multi-item list (e.g., `[0.01, 0.005]`).
*   **Note:** The pipeline performs a full Cartesian product of all lists.
