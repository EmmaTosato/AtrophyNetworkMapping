# User Guide: Deep Learning Pipeline (Nested CV)

This guide provides step-by-step instructions on how to use, configure, and interpret the Deep Learning pipeline for the ANM Verona project.

## 1. Quick Start 🚀

## 1. Quick Start

### Option A: The "Wizard" (Recommended)
Use the benchmark runner to launch multiple experiments automatically.

```bash
python src/DL_analysis/training/run_benchmark.py --pairs AD_PSP --models resnet
```

**Arguments:**
*   `--pairs`: Which comparisons to run.
    *   Examples: `AD_PSP`, `AD_CBS`, `PSP_CBS`
    *   Magic word: `all` (Runs all 3 pairs)
*   `--models`: Which architectures to test.
    *   Choices: `resnet`, `vgg16`, `alexnet`
    *   Magic word: `all` (Runs all 3 models)
*   `--dry_run`: Prints what it *would* do without actually running (useful for checking).
*   `--test_mode`: Runs a super-fast dummy version (1 epoch, few subjects) to check for bugs.

**Examples:**
```bash
# Run EVERYTHING (All pairs, all models) -> The "Weekend Run"
python src/DL_analysis/training/run_benchmark.py --pairs all --models all

# Run only VGG16 for AD vs PSP
python src/DL_analysis/training/run_benchmark.py --pairs AD_PSP --models vgg16
```

### Option B: The "Surgeon" (Manual Single Run)
Use the core script if you want to run exactly **ONE** specific experiment with granular control.

```bash
python src/DL_analysis/training/nested_cv_runner.py --group1 AD --group2 PSP --model resnet --output_dir my_test_folder
```

**Required Arguments:**
*   `--group1`: First Diagnostic Label (AD, PSP, CBS, CN).
*   `--group2`: Second Diagnostic Label.
*   `--model`: Model architecture (`resnet`, `vgg16`, `alexnet`).
*   `--output_dir`: Where to save the results.

## 2. Architecture Overview

The pipeline implements a **Nested Cross-Validation (5x5)** to ensure rigorous, unbiased evaluation.

*   **Outer Loop (5 Folds)**: Isolates 20% of data as a **Test Set** (never touched during training).
*   **Inner Loop (5 Folds)**: Uses the remaining 80% to find optimal hyperparameters (Learning Rate, etc.).
*   **Full Retrain**: Once hyperparameters are chosen, the model is retrained on the full Outer Train set (split 90/10 for internal validation).

**What it uses:**
*   **Data**: `data/FCmaps_processed` (Input .npy maps).
*   **Config**: `results/DL/global_experiment_config.json` (Search grids).
*   **Scripts**:
    *   `src/DL_analysis/training/nested_cv_runner.py`: The core engine.
    *   `src/DL_analysis/training/run_benchmark.py`: The automation wrapper.

## 3. Results & Interpretation

All results are stored in `results/DL/` organized by **Comparison > Model**.

Structure:
```
results/DL/
└── AD_PSP/
    └── resnet/
        ├── aggregated_results.json  <-- IL REPORT FINALE
        ├── fold_1/                  <-- Outer Fold 1 (Esperimento Indipendente)
        │   ├── metrics.json         <-- Risultato Test Set
        │   ├── history.json         <-- Curve di Training
        │   ├── plots/               <-- Grafici PNG
        │   └── models/              <-- Checkpoint (.pt)
        └── ...
```

### How to Read the Files

#### A. `aggregated_results.json` (The Truth)
Contains the **mean performance** on the Test Sets.
*   `mean_accuracy`: The most important metric (e.g., `0.63` = 63%).
*   `best_params`: The hyperparameters selected by the Inner Loop for each fold.

#### B. `history.json` & Plots (The Process)
Shows how the model learned during the **Full Retrain** phase.
*   **Train Curve**: How well it memorized the training data.
*   **Val Curve**: How well it generalized to the internal validation split (10%).
*   *Note*: These do NOT show the Test Set accuracy.

#### C. `metrics.json` (The Exam)
The specific score obtained on the isolated Test Set for that specific fold.

## 4. Configuration

To modify the search space (e.g., change Learning Rate ranges), edit:
**`src/DL_analysis/config/nested_cv_config.json`**

Example format:
```json
"resnet": {
    "lr": [0.01, 0.001],
    "batch_size": [8],
    ...
}
```
*After editing, relaunch the benchmark script to apply changes.*
