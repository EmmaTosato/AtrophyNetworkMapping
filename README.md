# ANM_Verona

**ANM_Verona** is a research framework for neuroimaging analysis, designed to classify and regress clinical variables from functional connectivity (FC) maps. It provides two parallel pipelines: a **Machine Learning (ML)** pipeline based on scikit-learn (using voxel-wise or network-based features) and a **Deep Learning (DL)** pipeline using 3D CNNs (ResNet, VGG, AlexNet) implemented in PyTorch.

## Features

- **Preprocessing Pipeline**: Automates thresholding, masking, and feature extraction from NIfTI files.
- **ML Analysis**:
  - **Classification**: Nested Cross-Validation (5x5) with Grid Search (RandomForest, GBM, KNN).
  - **Regression**: OLS regression with UMAP embedding and shuffling-based permutation testing.
- **DL Analysis**:
  - 3D CNN training with data augmentation.
  - Nested CV for robust performance estimation.
  - Supports multiple model architectures (ResNet, VGG).

## Prerequisites

- **OS**: Linux
- **Python**: 3.8+
- **Data**:
  - Functional Connectivity Maps (NIfTI `.nii.gz` or Numpy `.npy`).
  - Metadata CSVs containing subject IDs, groups (AD, PSP, CBS), and clinical scores (CDR_SB, MMSE).

### Dependencies
The project relies on the standard scientific Python stack. Key libraries include:
- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `statsmodels`
- `torch` (PyTorch)
- `nibabel` (Neuroimaging)
- `matplotlib`, `seaborn` (Plotting)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ANM_Verona
   ```

2. Set up `PYTHONPATH` to include `src`:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

## Quickstart

### 1. Run Machine Learning Classification
Executes the batch classification pipeline (Networks and Voxel+UMAP) across all defined pairs (AD vs PSP, AD vs CBS, etc.).

```bash
python src/ML_analysis/analysis/run_all_classifications.py
```
**Output**: Results are saved in `results/ML/`, aggregated in `total_aggregated_results.csv`.

### 2. Run Machine Learning Regression
Runs batch regression for CDR_SB and MMSE targets using voxel and network features.

```bash
python src/ML_analysis/analysis/run_all_regressions.py
```
**Output**: Diagnostic plots and statistics in `results/ML/voxel/` or `results/ML/networks/`.

### 3. Run Deep Learning Training
Runs the 3D CNN benchmark as defined in `src/DL_analysis/config/cnn.json`.

```bash
python src/DL_analysis/training/run_all.py --config src/DL_analysis/config/cnn.json
```
**Output**: Models and training logs in `results/DL/`.

## Project Structure

```text
src/
├── DL_analysis/       # Deep Learning Pipeline (CNNs, PyTorch)
│   ├── training/      # Training scripts (run_all.py, run_nested_cv.py)
│   ├── cnn/           # Model definitions and Dataset classes
│   └── config/        # JSON configs for experiments and grids
├── ML_analysis/       # Machine Learning Pipeline (sklearn)
│   ├── analysis/      # Core logic (classification.py, regression.py)
│   ├── config/        # JSON configs (ml_config.json)
│   └── loading/       # Data loading utilities
├── preprocessing/     # Feature extraction (processflat.py, process3d.py)
└── utils/             # Helper scripts
```

## Documentation

For more detailed information, see the full documentation:

- [Architecture Overview](doc/architecture.md): Detailed system pipelines and data flow.
- [CLI Reference](doc/cli.md): Command-line arguments and script usage.
