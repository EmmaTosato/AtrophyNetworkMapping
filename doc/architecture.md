# System Architecture

## Overview
The **ANM_Verona** framework consists of three main pipelines: Preprocessing (Data Preparation), Machine Learning (sklearn-based), and Deep Learning (PyTorch-based). The system is designed to classify and regress clinical variables (e.g., CDR_SB, MMSE) from functional connectivity (FC) maps.

## 1. Data Processing Layer

### A. Preprocessing
**Entrypoints**: `src/preprocessing/process3d.py`, `src/preprocessing/processflat.py`

Responsible for converting raw neuroimaging data into analysis-ready formats.

- **Inputs**: Raw NIfTI files (`.nii.gz`) and cognitive metadata (Excel/CSV).
- **Core Logic**:
  - **Thresholding**: Applies intensity thresholds (e.g., 0.1, 0.2) to remove weak connections.
  - **Masking**: Applies Gray Matter (GM) or Harvard-Oxford masks.
  - **Clustering**: Uses Gaussian Mixture Models (GMM) to cluster clinical scores into severity classes.
- **Outputs**:
  - **3D**: Processed `.npy` files in `data/FCmaps_processed`.
  - **Tabular**: Pickled DataFrames (`df_gm.pkl`, `df_har.pkl`) for ML.

### B. Augmentation
**Entrypoint**: `src/augmentation/augmentation.py`

Generates synthetic FC maps to increase dataset size or regularization for Deep Learning.

- **Dependencies**: Requires **FSL** (`fslmerge`, `fslmaths`) installed on the system.
- **Method**: Merges a patient's map with a random subset of healthy control (HCP) maps.
- **Outputs**: New `.nii.gz` files in `data/FCmaps_augmented/` and a tracking CSV.

## 2. Machine Learning Pipeline
**Location**: `src/ML_analysis/`

### A. Classification
Comparative analysis between subject groups (e.g., AD vs PSP).

- **Orchestrator**: `run_all_classifications.py`
- **Method**: Nested Cross-Validation (5x5) with Grid Search (RandomForest, GBM, KNN).
- **Features**: Supports both **Voxel-wise** (with UMAP reduction) and **Network-based** features.
- **Validation**: Permutation testing (1000 shuffles) for statistical significance.
- **Outputs**: `total_aggregated_results.csv`, Confusion Matrices.

### B. Regression
Predicts clinical scores (MMSE, CDR_SB).

- **Orchestrator**: `run_all_regressions.py`
- **Method**: Ordinary Least Squares (OLS) regression with UMAP embedding.
- **Validation**: "Shuffling regression" (null distribution generation).
- **Outputs**: RMSE statistics, Residual Plots, Actual vs Predicted Scatter plots.

### C. Clustering
Unsupervised subtype discovery.

- **Entrypoint**: `src/ML_analysis/analysis/clustering.py`
- **Method**: K-Means clustering on UMAP embeddings.
- **Outputs**: Cluster labels (`labels_km`) added to metadata.

## 3. Deep Learning Pipeline
**Location**: `src/DL_analysis/`
**Entrypoint**: `run_all.py`

Implements end-to-end 3D Deep Learning using PyTorch.

- **Models**: 3D CNNs (ResNet, VGG16, AlexNet).
- **Workflow**:
  1.  **Config**: Defined in `cnn.json`.
  2.  **Dataset**: Loads processed `.npy` volumes (Original + Augmented).
  3.  **Training**: Nested CV with Adam/SGD optimizers and Learning Rate schedulers.
  4.  **Early Stopping**: Prevents overfitting using validation set.
- **Outputs**:
  - Model Checkpoints (`.pt`).
  - Training History (`history.json`).
  - Per-fold and Aggregated Metrics (`nested_cv_results.csv`).

## Project Dependency Map

```text
[Raw NIfTI] --> [Preprocessing] --> [Processed NPY/PKL]
                     ^
                     | (Augmentation)
                     v
             [Augmented NIfTI/NPY]

      +------------------------+--------------------------+
      |                        |                          |
 [ML Pipeline]           [DL Pipeline]            [Clustering]
 (sklearn)               (PyTorch 3D CNN)         (Unsupervised)
      |                        |                          |
      +-> Classification       +-> ResNet/VGG             +-> K-Means
      +-> Regression           +-> Nested CV
```
