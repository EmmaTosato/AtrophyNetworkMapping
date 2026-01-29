# CLI Reference

## Machine Learning

### `run_all_classifications.py`
High-level orchestrator for batch classification.

- **Usage**: `python src/ML_analysis/analysis/run_all_classifications.py`
- **Behavior**: Runs defined comparisons (AD vs PSP, etc.) for both Voxel (UMAP) and Network datasets. Generates `summary_results.csv` and `total_aggregated_results.csv`.

### `run_all_regressions.py`
Batch runner for regression analysis (MMSE, CDR_SB).

- **Usage**: `python src/ML_analysis/analysis/run_all_regressions.py`
- **Configuration**: controlled by variables inside the script (`DATASETS`, `TARGETS`).

### `clustering.py`
Runs unsupervised K-Means clustering.

- **Usage**: `python src/ML_analysis/analysis/clustering.py`
- **Configuration**: controlled by `src/ML_analysis/config/ml_config.json`.
- **Outputs**: Updates metadata with `labels_km`.

---

## Individual ML Scripts (Manual Mode)

While the "run_all" scripts are recommended for batches, you can execute the core analysis scripts individually.
**Note**: These scripts generally do not accept command-line arguments (except `permutation_test.py`). Instead, they read parameters directly from `src/ML_analysis/config/ml_config.json`.

### `classification.py`
Runs a single classification experiment based on the current config.

1. Edit `src/ML_analysis/config/ml_config.json`:
   ```json
   "job": { "dataset_type": "networks" },
   "classification": { "group1": "AD", "group2": "CN", ... }
   ```
2. Run:
   ```bash
   python src/ML_analysis/analysis/classification.py
   ```

### `regression.py`
Runs a single regression experiment.

1. Edit `src/ML_analysis/config/ml_config.json`:
   ```json
   "regression": { "target_variable": "CDR_SB", ... }
   ```
2. Run:
   ```bash
   python src/ML_analysis/analysis/regression.py
   ```

### `permutation_test.py`
Runs standalone permutation testing for model validation using pre-calculated Best Params.

- **Usage**:
  ```bash
  python src/ML_analysis/analysis/permutation_test.py --config src/ML_analysis/config/permutation_config.json
  ```
- **Arguments**:
  - `--config`: JSON config defining comparisons and models.
  - `--suffix`: Optional suffix for the output filename.

### `clustering_evaluation.py`
*Note: This is a utility module imported by `clustering.py` and is not intended to be run directly as a script.*


---

## Deep Learning

### `run_all.py`
Main entrypoint for DL benchmarks.

**Usage**:
```bash
python src/DL_analysis/training/run_all.py --config src/DL_analysis/config/cnn.json
```

**Options**:
- `--dry_run`: Print commands without executing.
- `--grid_config`: Path to hyperparameter grid (default: `cnn_grid.json`).

**Note on Logging**:
For long runs, use `nohup` to direct output to a log file:
```bash
nohup python src/DL_analysis/training/run_all.py > benchmark.log 2>&1 &
```

---

## Preprocessing & Utils

### `process3d.py`
Prepares 3D NIfTI maps (Thresholding -> Masking -> Normalization).

- **Usage**: `python src/preprocessing/process3d.py`
- **Output**: `.npy` files in `data/FCmaps_processed`.

### `processflat.py`
Processes tabular data and aligns it with imaging data.

- **Usage**: `python src/preprocessing/processflat.py`

### `augmentation.py`
Generates augmented maps by merging patient maps with healthy controls.
**Requirement**: FSL (`fslmerge`, `fslmaths`) must be installed and in the PATH.

**Usage**:
```bash
python src/augmentation/augmentation.py \
  --subject_id <ID> \
  --dataset_dir <PATH> \
  --hcp_list <PATH_TO_HCP_LIST> \
  --output_dir <OUTPUT_PATH> \
  --csv_out <TRACKING_CSV> \
  --n_augmentations 10 \
  --subset_size 17
```
