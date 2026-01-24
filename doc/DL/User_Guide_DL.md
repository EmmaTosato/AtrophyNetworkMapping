# Deep Learning Pipeline: User Guide

## 1. Execution (How to Run)

The pipeline is **config-driven**. You define the experiments in a JSON file and launch a single automation script.

### Step 1: Define Jobs
Edit `src/DL_analysis/config/cnn.json` to define experiments (WHAT to run).

### Step 2: Configure Hyperparameters
Edit `src/DL_analysis/config/cnn_grid.json` to set search spaces (HOW to train).

*(See [DL_Configuration_Reference.md](DL_Configuration_Reference.md) for details)*

### Step 3: Launch
Run the automation script to execute all jobs sequentially.

```bash
# Standard Launch
python src/DL_analysis/training/run_all.py

# Check configuration execution plan (Dry Run)
python src/DL_analysis/training/run_all.py --dry_run
```

## 2. Architecture Overview

To ensure robust results, we use **Nested Cross-Validation (5x5)**.
*(See [Nested_CV.md](Nested_CV.md) for full technical logic)*

1.  **Outer Loop (5 Folds)**: Isolates 20% of subjects as a pure **Test Set**. These subjects are *never* seen during training or tuning.
2.  **Inner Loop (5 Folds)**: Uses the remaining 80% to run **Grid Search**. We train models on different hyperparameter combinations to find the best configuration.
3.  **Full Retrain**: Once the best hyperparameters are found (e.g., LR=0.01), we retrain a fresh model on the *entire* Outer Train set (using **Data Augmentation** to multiply the training data).
4.  **Final Evaluation**: This retrained model is evaluated on the specific Outer Test set.

## 3. Results & Output Interpretation

Results are organized hierarchically: `results/DL/[Group1]_[Group2]/[Model]/`

### Key Files

*   **`aggregated_results.json`** (The Main Report)
    *   Contains the **Mean Accuracy** and Standard Deviation across the 5 output folds.
    *   Use this file to compare model performance (e.g., "ResNet achieved 85% ± 3%").

*   **`fold_X/metrics.json`**
    *   The specific scores (Accuracy, AUC, F1) for that specific fold.
    *   Useful to spot if one specific data split yielded outlier results.

*   **`fold_X/plots/`**
    *   **`loss_curve.png`**: Inspect this to check for overfitting. If the validation loss (orange) goes up while training loss (blue) goes down, the model is overfitting.

*   **`fold_X/best_params.json`**
    *   Shows which hyperparameters won the Inner Grid Search for this fold.
