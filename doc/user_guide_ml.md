
# Machine Learning Pipeline Guide

## Overview
This suite handles traditional ML tasks: Classification (Voxel & Networks), Regression, and Clustering.

## 1. Classification Pipeline
Runs models (RandomForest, KNN, GradientBoosting) on both Voxel maps (using UMAP reduction) and Network outputs.

### How to Launch
To run all classification experiments:
```bash
python3 src/ML_analysis/analysis/run_all_classifications.py
```

### Components Used
*   **Orchestrator**: `src/ML_analysis/analysis/run_all_classifications.py` - Iterates through defined model/group configurations.
*   **Worker Script**: `src/ML_analysis/analysis/classification.py` - Runs the actual training and evaluation using the updated config.
*   **Configuration**: `src/ML_analysis/config/ml_config.json` - Dynamically updated by the orchestrator.

### Actions Performed
1.  Loads configuration presets (e.g., AD vs CBS, Voxel vs Networks).
2.  Updates `ml_config.json` for the specific run.
3.  Executes `classification.py` which performs:
    *   Feature loading (Voxel or Networks).
    *   UMAP reduction (if Voxel).
    *   Nested Cross-Validation.
    *   Permutation testing for significance (p-values).

### Results and Interpretation
Located in: `results/ML/networks/classification` OR `results/ML/voxel/umap_classification`.

*   **total_aggregated_results.csv**: Global summary of all comparison pairs.
*   **[Pair]/summary_results.csv**: Summary for a specific pair (e.g., AD_vs_CBS).
*   **[Pair]/[Model]/nested_cv_results.csv**: Detailed fold-by-fold metrics.
*   **permutation_stats.csv**: Statistical significance (p-values) for the model performance.

## 2. Regression Pipeline
Predicts clinical scores (e.g., MMSE, CDR-SB) from imaging data using OLS and Shuffling Regression.

### How to Launch
```bash
python3 src/ML_analysis/analysis/regression.py
```

### Components Used
*   **Script**: `src/ML_analysis/analysis/regression.py`.
*   **Input**: Metadata (clinical scores) and Imaging Data.

### Actions Performed
1.  Cleans data (removes missing target values).
2.  Projects features (UMAP) if configured.
3.  Fits an OLS model (Ordinary Least Squares).
4.  Runs Shuffling Regression (randomizes targets to build a null distribution for significance testing).

### Results and Interpretation
Located in: `results/ML/[dataset_type]/[task_type]/[Target_Variable]`.

*   **Diagnostic Plots**: Actual vs Predicted, Residuals.
*   **Console Output**: R-squared, RMSE, MAE, and Shuffled p-value.
*   **RMSE Stats**: Breakdown of error by subject group.

## 3. Clustering Pipeline
Performs unsupervised clustering (K-Means) to find data-driven subtypes using UMAP features.

### How to Launch
```bash
python3 src/ML_analysis/analysis/clustering.py
```

### Components Used
*   **Script**: `src/ML_analysis/analysis/clustering.py`.
*   **Algorithm**: K-Means (default n=3).

### Actions Performed
1.  Generates UMAP embedding of the data.
2.  Clusters the embedding using K-Means.
3.  Evaluates clusters against known groups or clinical scores.
4.  Updates metadata with new cluster labels.

### Results and Interpretation
Located in: `results/ML/voxel/umap_clustering` (typically).

*   **Plots**: Scatter plots of UMAP embeddings colored by Cluster vs True Group.
*   **Metadata**: The script updates the main metadata CSV with new cluster labels (e.g., `labels_km`).
