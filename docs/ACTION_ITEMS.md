# Action Items: Next Steps for ML/DL Alignment

## Executive Summary

Questo documento fornisce una lista prioritizzata di azioni da intraprendere per allineare e migliorare i pipeline ML e DL.

---

## 🚨 Critical Issues (Fix ASAP)

### 1. DL Seed Management
**Problem**: Seed controlla sia fold splits che weight initialization
**Current**:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Different per seed
torch.manual_seed(seed)  # Same seed
```

**Impact**: Different seeds → different train/val splits → not comparable
**Fix**:
```python
# In run.py::main_worker()
FIXED_FOLD_SEED = 42  # Same for all runs
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FIXED_FOLD_SEED)
torch.manual_seed(params['seed'])  # Variable seed for weights
```

**Priority**: 🔴 HIGH
**Estimated time**: 10 min
**File**: `src/DL_analysis/training/run.py`, line ~195

---

### 2. DL Missing AUC-ROC Metric
**Problem**: DL computes Acc, Prec, Rec, F1 but not AUC-ROC (ML has it)
**Current**: Only in `compute_metrics()`
**Fix**:
```python
# In test.py::compute_metrics()
from sklearn.metrics import roc_auc_score

def compute_metrics(y_true, y_pred, y_proba=None):
    # ... existing code ...
    if y_proba is not None:
        auc_roc = roc_auc_score(y_true, y_proba[:, 1])  # Assuming binary
        metrics["auc_roc"] = auc_roc
    return metrics
```

**Priority**: 🔴 HIGH
**Estimated time**: 20 min
**Files**: 
- `src/DL_analysis/testing/test.py`, line ~35
- `src/DL_analysis/training/run.py`, line ~330 (modify `evaluate()` call)

---

### 3. DL Results Aggregation
**Problem**: No automatic mean ± std across seeds (ML has it)
**Current**: Manual inspection of `all_testing_results.csv`
**Fix**: Create post-processing script
```python
# New file: src/DL_analysis/utils/aggregate_results.py
import pandas as pd
import numpy as np

def aggregate_testing_results(csv_path):
    df = pd.read_csv(csv_path)
    grouped = df.groupby('group')
    
    summary = grouped.agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    })
    
    print(f"Accuracy: {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    return summary
```

**Priority**: 🟠 MEDIUM
**Estimated time**: 30 min
**New file**: `src/DL_analysis/utils/aggregate_results.py`

---

## ⚠️ Important Improvements

### 4. Unified Output Format
**Problem**: ML and DL use different CSV formats
**ML CSV**:
```csv
seed,model,accuracy,precision,recall,f1,auc_roc
42,RF,0.653,0.650,0.645,0.647,0.689
```

**DL CSV**:
```csv
run_id,group,seed,accuracy,precision,recall,f1
run1,PSP vs CBS,42,0.702,0.695,0.700,0.697
```

**Fix**: Standardize to common format
```csv
experiment,seed,model,group,accuracy,precision,recall,f1,auc_roc
ml_rf,42,RandomForest,PSP vs CBS,0.653,0.650,0.645,0.647,0.689
dl_densenet,42,DenseNet3D,PSP vs CBS,0.702,0.695,0.700,0.697,0.710
```

**Priority**: 🟠 MEDIUM
**Estimated time**: 1 hour
**Files**: 
- `src/analysis/classification.py`
- `src/DL_analysis/utils/cnn_utils.py`

---

### 5. ML Feature Importance Analysis
**Problem**: RF feature importance not saved systematically
**Fix**:
```python
# In classification.py after training RF
if model_name == "RandomForest":
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    save_path = os.path.join(output_dir, f"feature_importance_seed{seed}.csv")
    feature_importance.to_csv(save_path, index=False)
```

**Priority**: 🟠 MEDIUM
**Estimated time**: 20 min
**File**: `src/analysis/classification.py`

---

### 6. DL Grad-CAM Visualization
**Problem**: No interpretability tool for DL (ML has feature importance)
**Fix**: Implement Grad-CAM for saliency maps
```python
# New file: src/DL_analysis/utils/gradcam.py
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)
    return grayscale_cam
```

**Priority**: 🟢 LOW (nice to have)
**Estimated time**: 2 hours
**New file**: `src/DL_analysis/utils/gradcam.py`

---

## 📊 Analysis and Comparison

### 7. Side-by-Side Results Notebook
**Problem**: No unified view of ML vs DL results
**Fix**: Create Jupyter notebook
```python
# notebooks/ml_vs_dl_comparison.ipynb

# Load results
ml_results = pd.read_csv('results/ml_analysis/classification_results.csv')
dl_results = pd.read_csv('results/runs/all_testing_results.csv')

# Aggregate DL by seed
dl_agg = aggregate_dl_results(dl_results)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_metric_comparison(ml_results, dl_agg, metric='accuracy', ax=axes[0])
plot_metric_comparison(ml_results, dl_agg, metric='precision', ax=axes[1])
plot_metric_comparison(ml_results, dl_agg, metric='f1', ax=axes[2])
```

**Priority**: 🟠 MEDIUM
**Estimated time**: 1 hour
**New file**: `notebooks/ml_vs_dl_comparison.ipynb`

---

### 8. Statistical Significance Testing
**Problem**: No formal test if ML vs DL difference is significant
**Fix**:
```python
from scipy.stats import ttest_ind

# Paired t-test (same seeds)
ml_accuracies = [0.653, 0.651, 0.655, 0.650, 0.654]  # 5 seeds
dl_accuracies = [0.702, 0.698, 0.705, 0.700, 0.703]  # 5 seeds

t_stat, p_value = ttest_ind(ml_accuracies, dl_accuracies)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("DL is significantly better than ML")
```

**Priority**: 🟠 MEDIUM
**Estimated time**: 30 min
**File**: Add to `notebooks/ml_vs_dl_comparison.ipynb`

---

## 🔬 Advanced Features

### 9. Ensemble ML + DL
**Problem**: Not leveraging complementary strengths
**Fix**: Combine predictions via voting or stacking
```python
# Soft voting
ml_proba = rf_model.predict_proba(X_test)
dl_proba = softmax(dl_model(X_test_tensor))

ensemble_proba = (ml_proba + dl_proba) / 2
ensemble_pred = np.argmax(ensemble_proba, axis=1)
```

**Priority**: 🟢 LOW (research)
**Estimated time**: 3 hours
**New file**: `src/analysis/ensemble.py`

---

### 10. Transfer Learning for DL
**Problem**: Training from scratch on small dataset
**Fix**: Use ImageNet pretrained weights
```python
# In models.py::ResNet3D
from torchvision.models.video import r3d_18

class ResNet3D(nn.Module):
    def __init__(self, n_classes, in_channels=1, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = r3d_18(weights='DEFAULT')  # ImageNet pretrained
        else:
            self.model = r3d_18(weights=None)
        
        # Modify stem for single channel (requires weight adaptation)
        # ... existing code ...
```

**Priority**: 🟢 LOW (experimental)
**Estimated time**: 4 hours
**File**: `src/DL_analysis/cnn/models.py`

---

## 📝 Documentation Updates

### 11. Update Empty MD Files
**Problem**: ML_CLASSIFICATION_DOCUMENTATION.md and QUICK_REFERENCE.md are empty
**Fix**: Consolidate content from other docs
```bash
# Copy relevant sections from:
# - TECHNICAL_GUIDE.md → ML_CLASSIFICATION_DOCUMENTATION.md
# - QUICK_COMMANDS.md → QUICK_REFERENCE.md
```

**Priority**: 🟢 LOW
**Estimated time**: 30 min
**Files**: 
- `docs/ML_CLASSIFICATION_DOCUMENTATION.md`
- `docs/QUICK_REFERENCE.md`

---

### 12. Add Code Comments
**Problem**: Some functions lack docstrings
**Fix**: Add comprehensive docstrings
```python
def run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold):
    """
    Train model for fixed number of epochs with validation after each.
    
    Tracks best model based on validation accuracy (tie-breaker: loss).
    Saves checkpoint, plots learning curves, and logs per-epoch metrics.
    
    Args:
        model (nn.Module): Neural network to train
        train_loader (DataLoader): Training data batches
        val_loader (DataLoader): Validation data batches
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        optimizer (torch.optim.Optimizer): Optimizer (Adam/SGD)
        params (dict): Configuration dict with keys:
            - epochs (int): Number of training epochs
            - device (str): 'cuda' or 'cpu'
            - plot (bool): Whether to save learning curves
            - actual_run_dir (str): Output directory
        fold (int): Current fold number (for logging)
    
    Returns:
        Tuple[float, float, float, int]: 
            - best_accuracy: Best validation accuracy achieved
            - best_train_loss: Training loss at best epoch
            - best_val_loss: Validation loss at best epoch
            - best_epoch: Epoch number of best model
    
    Side Effects:
        - Saves checkpoint to params['ckpt_path_evaluation']
        - (Optional) Saves plots to params['actual_run_dir']/plots/
        - (Optional) Appends to Excel file training_folds.xlsx
    
    Example:
        >>> best_acc, train_loss, val_loss, epoch = run_epochs(
        ...     model, train_loader, val_loader, criterion, optimizer, 
        ...     params={'epochs': 50, 'device': 'cuda', ...}, fold=1
        ... )
        >>> print(f"Best accuracy: {best_acc:.3f} at epoch {epoch}")
    """
    # ... implementation ...
```

**Priority**: 🟢 LOW
**Estimated time**: 2 hours
**Files**: All `.py` files in `src/DL_analysis/`

---

## 🧪 Testing and Validation

### 13. Unit Tests
**Problem**: No automated testing
**Fix**: Create test suite
```python
# New file: test/test_dl_functions.py
import pytest
from src.DL_analysis.training.train import train, validate

def test_train_single_epoch():
    # Mock model, loader, criterion, optimizer
    loss = train(mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu')
    assert isinstance(loss, float)
    assert loss > 0

def test_validate_returns_tuple():
    result = validate(mock_model, mock_loader, mock_criterion, 'cpu')
    assert isinstance(result, tuple)
    assert len(result) == 2
```

**Priority**: 🟢 LOW
**Estimated time**: 4 hours
**New file**: `test/test_dl_functions.py`

---

### 14. Integration Test
**Problem**: No end-to-end test
**Fix**: Small dataset test
```bash
# Create mini dataset (5 train, 2 test)
# Run complete pipeline
python run_train.py --config test_config.json --epochs 2

# Verify output files exist
assert os.path.exists('results/test_runs/all_training_results.csv')
```

**Priority**: 🟢 LOW
**Estimated time**: 2 hours
**New file**: `test/test_integration.sh`

---

## 📈 Performance Optimization

### 15. DL Training Speed
**Problem**: 5 folds × 50 epochs = 3 hours per seed
**Optimizations**:
1. **Mixed precision training** (AMP): 1.5-2× speedup
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, labels)
   ```

2. **DataLoader num_workers**: Faster data loading
   ```python
   DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
   ```

3. **Early stopping**: Stop if no improvement
   ```python
   if epoch - best_epoch > patience:
       break  # Stop training
   ```

**Priority**: 🟢 LOW (optimization)
**Estimated time**: 3 hours
**File**: `src/DL_analysis/training/train.py`

---

## 🎯 Priority Summary

### Must Do (Next Week)
1. ✅ Fix DL seed management (fold split vs init)
2. ✅ Add AUC-ROC to DL
3. ✅ Create DL results aggregation script

### Should Do (Next Month)
4. Unified output CSV format
5. ML feature importance saving
6. Side-by-side comparison notebook
7. Statistical significance testing

### Nice to Have (Future)
8. DL Grad-CAM visualization
9. Ensemble ML + DL
10. Transfer learning
11. Documentation updates
12. Unit tests
13. Performance optimization

---

## 📋 Implementation Checklist

### Week 1
- [ ] Fix DL seed management (Issue #1)
- [ ] Add AUC-ROC metric to DL (Issue #2)
- [ ] Create `aggregate_results.py` (Issue #3)
- [ ] Test changes with single run

### Week 2
- [ ] Implement unified CSV format (Issue #4)
- [ ] Add ML feature importance saving (Issue #5)
- [ ] Create comparison notebook (Issue #7)
- [ ] Run statistical tests (Issue #8)

### Week 3
- [ ] Implement Grad-CAM (Issue #6)
- [ ] Experiment with ensemble (Issue #9)
- [ ] Update empty docs (Issue #11)

### Week 4
- [ ] Add comprehensive docstrings (Issue #12)
- [ ] Create unit tests (Issue #13)
- [ ] Performance profiling (Issue #15)
- [ ] Final validation and cleanup

---

## 🚀 Quick Start Guide for Issues

### To Fix Issue #1 (Seed Management)
```bash
cd /data/users/etosato/ANM_Verona
# Edit src/DL_analysis/training/run.py
# Line ~195: Change random_state=params['seed'] to random_state=42
git diff src/DL_analysis/training/run.py  # Verify change
python src/DL_analysis/training/run_train.py  # Test
```

### To Fix Issue #2 (AUC-ROC)
```bash
# Edit src/DL_analysis/testing/test.py
# Add: from sklearn.metrics import roc_auc_score
# Modify compute_metrics() to include AUC-ROC
# Edit src/DL_analysis/training/run.py
# Modify evaluate() call to return probabilities
python src/DL_analysis/testing/run_test.py  # Test
```

### To Fix Issue #3 (Aggregation)
```bash
# Create new file
touch src/DL_analysis/utils/aggregate_results.py
# Implement aggregate_testing_results() function
python -c "
from src.DL_analysis.utils.aggregate_results import aggregate_testing_results
aggregate_testing_results('results/runs/all_testing_results.csv')
"
```

---

**Last updated**: 2025-01-19 17:00 CET
**Total estimated time**: ~30 hours (spread over 4 weeks)
