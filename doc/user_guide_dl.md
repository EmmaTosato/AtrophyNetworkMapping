
# Deep Learning Pipeline Guide

## Overview
This pipeline trains and evaluates 3D CNN models (ResNet, AlexNet, VGG16) on ANM voxel data. It handles nested cross-validation, training, testing, and result aggregation.

## 1. How to Launch
To run the full benchmark (all models, all pairs):

```bash
nohup python3 src/DL_analysis/training/run_all.py \
    --config src/DL_analysis/config/cnn.json \
    > logs/orchestration/benchmark_final.log 2>&1 &
```

### Components Used
*   **Orchestrator**: `src/DL_analysis/training/run_all.py` - Manages the queue of experiments.
*   **Single Experiment**: `src/DL_analysis/training/run_nested_cv.py` - Executes a single experiment (Pair + Model).
*   **Configuration**: `src/DL_analysis/config/cnn.json` (defines pairs/models) and `cnn_grid.json` (hyperparameters).

### Actions Performed
1.  Parses the configuration to identify which Models (e.g., AlexNet) and Pairs (e.g., AD vs CN) to run.
2.  Iterates through each combination.
3.  Launches a Nested Cross-Validation (5 folds).
4.  Trains the model, saves the best weights, and generates performance plots.

## 2. Inputs and Results

### Input Data
*   **Voxel Maps**: `/data/users/etosato/ANM_Verona/data/FCmaps_processed` (and augmented versions).
*   **Metadata**: `assets/metadata/labels.csv`.

### Output Location
Results are stored in `results/DL/` organized by **Comparison** -> **Model**.

Example structure:
```text
results/DL/
├── total_aggregated_results.csv        # Global Summary (Accuracy, AUC, etc. for all pairs)
├── AD_CBS/                             # Specific Comparison Folder
│   ├── summary_results.csv             # Summary for this pair
│   └── alexnet/                        # Model Folder
│       ├── aggregated_results.csv      # Mean/Std metrics
│       ├── nested_cv_results.csv       # Metrics for each fold (1-5)
│       └── fold_1/                     # Fold-specific artifacts
│           ├── models/best_model.pt    # Saved Model Weights
│           └── plots/                  # Training curves (Loss/Accuracy)
```

### How to Read Results
1.  **Global View**: Open `total_aggregated_results.csv` to compare performance across all experiment pairs and models.
2.  **Stability Check**: Check `nested_cv_results.csv` within a model folder to see variance across the 5 folds.
3.  **Debugging**: If a run fails, check the logs in `logs/DL_results/[Pair]/[Model]_run.log`.
