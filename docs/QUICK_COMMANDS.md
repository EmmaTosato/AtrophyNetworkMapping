q# Quick Reference: ML vs DL Commands

## Machine Learning

### Standard Training (Fixed Split, No Tuning)
```bash
cd /data/users/etosato/ANM_Verona
# Edit src/ML_analysis/config/ml_config.json:
# - "tuning": false
# - "permutation_test": true
# - "seeds": [42, 123, 2023, 31415, 98765]

python -c "
from src.analysis.classification import main_classification
from src.config.config_loader import ConfigLoader

config = ConfigLoader('src/ML_analysis/config/ml_config.json')
main_classification(config)
"
```

**Output**:
- `results/ml_analysis/classification_results.csv`
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Format: Mean ± Std across 5 seeds

---

### With Hyperparameter Tuning
```bash
# Edit ml_config.json:
# - "tuning": true

# Same command as above
```

**Output**: GridSearchCV finds best params, then tests on test set

---

## Deep Learning

### 1. Standard Training (Multiple Seeds)
```bash
cd /data/users/etosato/ANM_Verona

# Edit src/DL_analysis/config/cnn_config.json:
# - "crossval_flag": true
# - "evaluation_flag": false
# - "tuning_flag": false

python src/DL_analysis/training/run_train.py
```

**Output**:
- `results/runs/run{1-5}/` (one per seed)
- `results/runs/all_training_results.csv`
- Checkpoints: `best_model_fold{1-5}.pt` per run
- Excel: `training_folds.xlsx` (per-epoch metrics)

---

### 2. Testing Trained Models
```bash
# After training, test best fold checkpoints:

python src/DL_analysis/testing/run_test.py
```

**Output**:
- `results/runs/all_testing_results.csv`
- Confusion matrices in each run folder

---

### 3. Hyperparameter Tuning (Grid Search)
```bash
# Edit src/DL_analysis/config/cnn_grid.json:
# - Define grid parameters (model_type, lr, batch_size, etc.)
# - Set experiment: group1, group2, run_id

python src/DL_analysis/training/hyper_tuning.py
```

**Output**:
- `results/tuning/tuning{run_id}/grid_results.csv`
- One folder per config: `config{1-N}/`
- Each config has 5 fold checkpoints

---

### 4. Testing Best Tuning Configs
```bash
# Edit src/DL_analysis/testing/run_test_tuning.py:
# - Set csv_path to tuning results
# - (Optional) Filter by accuracy thresholds

python src/DL_analysis/testing/run_test_tuning.py
```

**Output**: Testing results for selected configs

---

## Common Operations

### Check GPU Availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Monitor GPU Usage (during DL training)
```bash
watch -n 1 nvidia-smi
```

### View Training Logs (DL)
```bash
# While training:
tail -f /data/users/etosato/ANM_Verona/results/runs/run1/log_train1

# After training:
less /data/users/etosato/ANM_Verona/results/runs/run1/log_train1
```

### View Results CSV
```bash
# ML results:
cat results/ml_analysis/classification_results.csv | column -t -s','

# DL training results:
cat results/runs/all_training_results.csv | column -t -s','

# DL testing results:
cat results/runs/all_testing_results.csv | column -t -s','
```

---

## File Paths Reference

### ML Files
```
src/ML_analysis/
├── config/
│   └── ml_config.json          # Main config
├── analysis/
│   └── classification.py       # Main script
└── utils/
    └── ml_utils.py             # Helper functions

results/ml_analysis/
└── classification_results.csv  # Output
```

### DL Files
```
src/DL_analysis/
├── config/
│   ├── cnn_config.json         # Main config
│   └── cnn_grid.json           # Tuning grid
├── training/
│   ├── run_train.py            # Multi-seed launcher
│   ├── run.py                  # Main pipeline
│   ├── train.py                # Train/validate
│   └── hyper_tuning.py         # Grid search
├── testing/
│   ├── run_test.py             # Batch testing
│   └── test.py                 # Metrics
├── cnn/
│   ├── models.py               # Architectures
│   └── datasets.py             # Data loaders
└── utils/
    └── cnn_utils.py            # Helpers

results/
├── runs/
│   ├── all_training_results.csv
│   ├── all_testing_results.csv
│   └── run{1-N}/               # Per-run folders
└── tuning/
    └── tuning{1-N}/            # Per-tuning folders
```

---

## Configuration Examples

### ML Config (ml_config.json)
```json
{
  "data": {
    "split_csv": "assets/split/ADNI_PSP_splitted.csv",
    "data_dir": "data/FCmaps_processed"
  },
  "training": {
    "tuning": false,
    "permutation_test": true,
    "seeds": [42, 123, 2023, 31415, 98765],
    "n_folds": 5
  },
  "models": {
    "RandomForest": {
      "n_estimators": [100, 200],
      "max_depth": [null, 10, 20]
    }
  }
}
```

### DL Config (cnn_config.json)
```json
{
  "training": {
    "model_type": "densenet",
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.001,
    "optimizer": "adam",
    "n_folds": 5,
    "seed": 42
  },
  "experiment": {
    "group1": "PSP",
    "group2": "CBS",
    "run_id": 1,
    "crossval_flag": true,
    "evaluation_flag": false
  }
}
```

---

## Troubleshooting

### ML: "UMAP data leakage" Warning
**Fix**: Already fixed in classification.py
- UMAP fit only on train, transform on test
- Removed `umap_all=True` parameter

### DL: CUDA Out of Memory
**Fix**:
```json
// In cnn_config.json, reduce batch_size:
"batch_size": 4  // Instead of 8 or 16
```

### DL: "Missing split CSV" Error
**Check**: 
```bash
ls assets/split_cnn/
# Should contain: ADNI_PSP_splitted.csv, PSP_CBS_splitted.csv, etc.
```

### DL: Checkpoints Not Loading
**Check**:
```python
# In run.py, ensure correct path:
config["experiment"]["ckpt_path_evaluation"] = "results/runs/run1/best_model_fold1.pt"
```

---

## Performance Benchmarks (Approximate)

| Task | ML | DL |
|------|----|----|
| Single seed training | ~10 sec | ~3 hours (5 folds × 50 epochs) |
| 5 seeds training | ~1 min | ~15 hours |
| Hyperparameter tuning | ~5 min | ~50 hours (18 configs) |
| Testing | <1 sec | ~5 min |
| Total experiment | ~15 min | ~20-50 hours |

**Hardware**: 
- ML: CPU (Intel Xeon, 16 cores)
- DL: GPU (NVIDIA V100, 16 GB VRAM)

---

## Next Steps

### For ML
1. [x] Fix data leakage issues
2. [x] Add test set evaluation
3. [ ] Add ensemble methods (voting, stacking)
4. [ ] Feature selection analysis
5. [ ] Add SHAP values for interpretability

### For DL
1. [x] Implement 5-fold CV
2. [x] Save per-epoch metrics
3. [ ] Fix seed management (fold split vs init)
4. [ ] Add AUC-ROC metric
5. [ ] Implement Grad-CAM visualization
6. [ ] Try transfer learning (ImageNet pretrained)

### For Both
1. [ ] Unified output format for comparison
2. [ ] Side-by-side results notebook
3. [ ] Statistical significance testing (ML vs DL)
4. [ ] Ensemble ML + DL predictions
5. [ ] Final manuscript figures
