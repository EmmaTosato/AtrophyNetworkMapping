# Augmentation Guide

## Overview
Generates "augmented" connectivity maps by merging a patient's map with connectivity maps from healthy controls (HCP dataset). This increases dataset size and variability, acting as a regularization technique for Deep Learning.

## 1. How to Launch
The augmentation process is computationally intensive and should be run via SLURM.

```bash
sbatch src/augmentation/augmentation.sh
```

### Components Used
*   **Launcher**: `src/augmentation/augmentation.sh` (SLURM script).
*   **Logic**: `src/augmentation/augmentation.py`.
*   **Input Data**: Raw map list defined in the dataset directory.

### Configuration
Edit variables in `src/augmentation/augmentation.sh`:

*   `N_AUG` (Default: 10): Number of new maps to generate per patient.
*   `SUBSET_SIZE` (Default: 17): Number of HCP subjects to average into the patient map.
    *   Lower size = More noise/variability.
    *   Higher size = Smoother, closer to group average.
*   `DATASET_DIR`: Path pointing to the folder containing the subject list.

### Actions Performed
For each patient (e.g., `Subj_001`):
1.  Selects `N_AUG` disjoint lists of `SUBSET_SIZE` HCP subjects.
2.  Merges `Subj_001` + `HCP_List` -> `Subj_001_aug`.
3.  Saves the tracking info (which HCP subjects were used) to ensure reproducibility.

## 2. Results
*   **Output Folder**: `data/FCmaps_augmented/`.
*   **Tracking File**: `data/aug_tracking.csv` (records HCP subjects used for each augmentation).
