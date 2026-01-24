# User Guide: ML Pipeline

This guide explains how to run the Machine Learning modules.
All commands must be executed from the project root: `/data/users/etosato/ANM_Verona`.

## 1. Classification

Implements a Nested Cross-Validation (5x5) pipeline to robustly evaluate classifiers on neuroimaging data.

### Supported Models
*   **RandomForest**: Ensemble of decision trees (standard baseline).
*   **GradientBoosting**: Boosting algorithm (e.g., XGBoost style) for high performance.
*   **KNN**: K-Nearest Neighbors (distance-based).

### How to Run

**A. Automatic (Recommended)**
Runs all experimental pairs (e.g., AD vs PSP, AD vs CBS) defined in `src/ML_analysis/analysis/run_all_classifications.py`.
```bash
python src/ML_analysis/analysis/run_all_classifications.py
```

**B. Manual (Single Run)**
Runs a single configuration based on `ml_config.json`.
1.  Edit `src/ML_analysis/config/ml_config.json`:
    *   `group1` / `group2`: Diagnostic groups (e.g., "AD", "PSP").
    *   `dataset_type`: "voxel" (masked maps) or "networks" (Yeo atlas).
2.  Run:
```bash
python src/ML_analysis/analysis/classification.py
```

### Interpretation of Results
Path: `results/ML/[dataset]/[umap]/[group1]_[group2]/`

*   **`nested_cv_summary.csv`**: The primary report. It contains the **Mean** and **Std** for:
    *   `Accuracy`: Overall correctness.
    *   `AUC_ROC`: Ability to distinguish classes (0.5 = random, 1.0 = perfect).
    *   `F1-Score`: Balance between Precision and Recall.
*   **`[ModelName]/confusion_matrix.png`**: Visualizes where the model makes mistakes (e.g., confusing PSP for AD).

---

## 2. Regression

Predicts continuous clinical scores (e.g., CDR_SB, MMSE) from brain connectivity.

### Workflow Details
1.  **Imputation**: Replaces missing features if flagged in config.
2.  **Transformation**: Can log-transform targets (`y_log_transform`) to normalize skewed clinical scores.
3.  **Covariates**: Optionally regresses out Age, Sex, or Education before analysis.
4.  **Shuffling Test**: Compares the model's R^2 against 100 random permutations to calculate a p-value.

### How to Run
1.  Edit `src/ML_analysis/config/ml_config.json`:
    *   `target_variable`: The score to predict (e.g., "CDR_SB").
    *   `flag_covariates`: Set `true` to control for confounders.
2.  Run:
```bash
python src/ML_analysis/analysis/regression.py
```

### Results
Path: `results/ML/[dataset]/[umap]/[target_variable]/`

*   **Log File**: Check this for the **Empirical p-value**. If p < 0.05, the prediction is significantly better than chance.
*   **`actual_vs_predicted.png`**: Points closer to the diagonal line indicate better predictions.

---

## 3. Clustering

Unsupervised analysis to discover data-driven subgroups (biotypes).

### Workflow Details
1.  **UMAP**: Nonlinear dimensionality reduction to map high-dim brain data to 2D.
2.  **K-Means**: partitions subjects into K groups.
3.  **HDBSCAN**: Identifies density-based clusters (good for noise rejection).
4.  **Enrichment**: Compares found clusters against clinical labels (AD/PSP/CBS) to check for biological relevance.

### How to Run
1.  Edit `src/ML_analysis/config/ml_config.json` (`umap: true`, `dataset_type: voxel`).
2.  Run:
```bash
python src/ML_analysis/analysis/clustering.py
```

### Results
Path: `results/ML/[dataset]/umap_classification/optimal_cluster/`

*   **`umap_embedding.png`**: The "map" of your subjects.
*   **`clusters_vs_groups.png`**: Bar chart showing if, for example, Cluster 1 gives a pure AD population or a mix.
