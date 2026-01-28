# Preprocessing Guide

## Overview
Utilities for processing 3D maps (masking, normalization) and tabular data.

## 1. 3D Map Processing
Used to prepare NIfTI files for analysis using `src/preprocessing/process3d.py`.

### How to Launch
```bash
python3 src/preprocessing/process3d.py
```

### Components Used
*   **Script**: `src/preprocessing/process3d.py`.
*   **Config**: Loaded via `ML_analysis.loading.config.ConfigLoader`. Ensure `config/main_config.yaml` (or equivalent) has the correct paths.

### Actions Performed
1.  **Loading**: Reads (`.nii.gz`) files from the input directory.
2.  **Masking**: Applies a Gray Matter (GM) mask to remove non-brain voxels.
3.  **Thresholding**: Sets weak connections (e.g., < 0.2) to zero to enforce sparsity (if configured).
4.  **Normalization**:
    *   *Min-Max*: Scales values between 0 and 1.
    *   *Z-Score*: (Optional) Subtracts mean, divides by std.

### Results
*   **Output**: Saved as `.npy` files in `data/FCmaps_processed` (or configured output dir).
*   **Format**: Numpy arrays ready for ML/DL loading.

## 2. Tabular Data Processing
Used for CSV/Excel handling using `processflat.py`.

### How to Launch
```bash
python3 src/preprocessing/processflat.py
```

### Components Used
*   **Script**: `src/preprocessing/processflat.py`.
*   **Input**: Raw clinical Excel/CSV files.

### Actions Performed
1.  **Alignment**: Ensures feature matrix rows align with metadata rows.
2.  **Cleaning**: Handles missing values and formatting.
3.  **GMM Clustering**: Can automatically group continuous scores (e.g., CDR_SB) into discrete severity classes (Low/Medium/High).
