# User Guide: Data Processing

Scripts for data manipulation and augmentation.

## 1. Data Augmentation

Generates "augmented" connectivity maps by merging a patient's map with connectivity maps from healthy controls (HCP dataset).
**Purpose**: Increases dataset size and variability, acting as a regularization technique for Deep Learning.

### Key Scripts
*   **Launcher**: `src/augmentation/augmentation.sh` (SLURM script).
*   **Core Logic**: `src/augmentation/augmentation.py`.

### Configuration Variables
To customize the process, edit `src/augmentation/augmentation.sh`:

*   `N_AUG` (Default: 10): How many new maps to generate per patient.
*   `SUBSET_SIZE` (Default: 17): How many HCP subjects to average into the patient map.
    *   *Lower size* = More noise/variability.
    *   *Higher size* = Smoother, closer to group average.

### How to Run
This process is computationally intensive. Always use SLURM.

1.  Open `src/augmentation/augmentation.sh`.
2.  Verify `DATASET_DIR` points to your raw maps (`.nii.gz`).
3.  Submit:
```bash
sbatch src/augmentation/augmentation.sh
```

### Output Logic
For each patient (e.g., `Subj_001`):
1.  Algorithm selects 10 disjoint lists of 17 HCP subjects.
2.  Merges `Subj_001` + `HCP_List_1` -> `Subj_001_aug_1`.
3.  Output is saved to `data/FCmaps_augmented/`.
4.  A `tracking.csv` file records exactly which HCP subjects were used for each augmentation to ensure reproducibility.

---

## 2. Preprocessing

Utilities for processing 3D maps and tabular data.

### `process3d.py`: 3D Map Processing
Used to prepare NIfTI files for analysis.
*   **Masking**: Applies a Gray Matter (GM) mask to remove non-brain voxels.
*   **Normalization**:
    *   *Z-Score*: Subtracts mean, divides by std (common for Deep Learning).
    *   *Min-Max*: Scales values between 0 and 1.
*   **Thresholding**: Sets weak connections (e.g., < 0.2) to zero to enforce sparsity.

### `processflat.py`: Tabular Data Processing
Used for CSV/Excel handling.
*   **Alignment**: Ensures row `i` in the feature matrix corresponds to row `i` in the metadata.
*   **GMM Clustering**: Can automatically group continuous scores (like CDR_SB) into discrete severity classes (Low/Medium/High) using Gaussian Mixture Models.
