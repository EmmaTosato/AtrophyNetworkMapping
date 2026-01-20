# Deep Learning Analysis - Complete Documentation

## 1. Directory Structure

```
src/DL_analysis/
├── config/
│   ├── cnn_config.json       # Main configuration file
│   └── cnn_grid.json         # Hyperparameter grid for tuning
├── training/
│   ├── run_train.py          # Launcher for multiple seeds
│   ├── run.py                # Main training/evaluation pipeline
│   ├── run_test_tuning.py    # Test best configs from tuning
│   ├── train.py              # Training and validation functions
│   └── hyper_tuning.py       # Grid search for hyperparameters
├── testing/
│   ├── run_test.py           # Batch testing launcher
│   └── test.py               # Testing and metrics functions
├── cnn/
│   ├── models.py             # Neural network architectures
│   └── datasets.py           # PyTorch Dataset classes
└── utils/
    └── cnn_utils.py          # Helper functions for summaries
```

---

## 2. Function Definitions by File

### 2.1 `training/run_train.py` (Lines 1-59)
**Purpose**: Launch multiple training runs with different random seeds

**No function definitions** - Script-level execution only

**Key Operations**:
- Loads base config from JSON
- Loops over seeds: `[42, 123, 2023, 31415, 98765]`
- Updates config with each seed and run_id
- Calls `run.py` via subprocess
- Merges results with seeds in final CSV

---

### 2.2 `training/run.py` (Lines 1-397)
**Purpose**: Main training and evaluation pipeline with cross-validation

#### Function: `set_seed(seed)` (Lines 28-38)
```python
def set_seed(seed):
    """Set all random seeds for reproducibility"""
```
- Sets random seeds for: Python, NumPy, PyTorch
- Disables CUDNN optimizations for determinism

#### Function: `run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold)` (Lines 41-121)
```python
def run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold):
    """Train for fixed epochs and validate after each epoch"""
```
- Tracks best model based on validation accuracy
- Saves checkpoint with best epoch
- Optionally plots learning curves
- Saves per-epoch metrics to Excel
- Returns: `best_accuracy, best_train_loss, best_val_loss, best_epoch`

#### Function: `main_worker(params, config_id=None)` (Lines 124-397)
```python
def main_worker(params, config_id=None):
    """Main training/evaluation with cross-validation or test mode"""
```
- **Cross-validation mode** (`crossval_flag=True`):
  - Uses `StratifiedKFold(n_splits=5)`
  - Trains on each fold with `run_epochs()`
  - Tracks best fold based on accuracy
  - Saves training summary to CSV
  - Returns best fold info if `tuning_flag=True`
  
- **Evaluation mode** (`evaluation_flag=True`):
  - Loads best checkpoint
  - Evaluates on test set
  - Computes metrics (accuracy, precision, recall, F1)
  - Saves testing summary to CSV
  - Plots confusion matrix

**Key Variables**:
- `best_fold_info`: Tracks best fold's accuracy and loss
- `fold_accuracies`: List of accuracies across folds
- `ckpt_path_evaluation`: Path to best model checkpoint

---

### 2.3 `training/train.py` (Lines 1-100)
**Purpose**: Core training and validation functions

#### Function: `train(model, train_loader, criterion, optimizer, device)` (Lines 5-36)
```python
def train(model, train_loader, criterion, optimizer, device):
    """Perform one epoch of training"""
```
- Sets model to training mode
- Forward pass → loss → backprop → optimizer step
- Returns: average training loss

#### Function: `validate(model, val_loader, criterion, device)` (Lines 38-71)
```python
def validate(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
```
- Sets model to evaluation mode
- Computes loss and accuracy without gradients
- Returns: `val_loss, val_accuracy`

#### Function: `plot_losses(train_losses, val_losses, val_accuracies=None, save_path=None, title=None)` (Lines 73-100)
```python
def plot_losses(train_losses, val_losses, val_accuracies=None, save_path=None, title=None):
    """Plot training/validation curves"""
```
- Plots train loss, val loss, optional val accuracy
- Saves to PNG file

---

### 2.4 `training/hyper_tuning.py` (Lines 1-103)
**Purpose**: Grid search for hyperparameter optimization

#### Function: `is_valid_combo(params)` (Lines 8-21)
```python
def is_valid_combo(params):
    """Check if hyperparameter combo is valid (GPU memory constraints)"""
```
- Enforces batch size constraints per model type:
  - VGG16: batch_size in [4, 8, 16]
  - ResNet/DenseNet: batch_size in [4, 16]

#### Function: `tuning(base_args_path, grid_path)` (Lines 23-103)
```python
def tuning(base_args_path, grid_path):
    """Perform grid search over hyperparameters"""
```
- Loads base config and grid parameters
- Generates all combinations with `itertools.product()`
- Filters invalid combos
- For each valid combo:
  - Calls `main_worker()` with `tuning_flag=True`
  - Collects results (best fold, avg accuracy, etc.)
- Saves all results to `grid_results.csv`

---

### 2.5 `training/run_test_tuning.py` (Lines 1-96)
**Purpose**: Run testing on best configurations from tuning

**No function definitions** - Script-level execution only

**Key Operations**:
- Loads tuning results CSV
- Filters by specific accuracy thresholds (optional)
- For each config:
  - Updates config JSON with best hyperparameters
  - Sets `evaluation_flag=True`, `crossval_flag=False`
  - Calls `run.py` via subprocess
- Merges results with tuning info

---

### 2.6 `testing/test.py` (Lines 1-110)
**Purpose**: Testing and metrics computation

#### Function: `evaluate(model, loader, device)` (Lines 7-32)
```python
def evaluate(model, loader, device):
    """Evaluate model on test set"""
```
- Sets model to evaluation mode
- Iterates through dataloader
- Returns: `y_true, y_pred` (NumPy arrays)

#### Function: `compute_metrics(y_true, y_pred)` (Lines 35-57)
```python
def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
```
- Computes: accuracy, precision, recall, F1, confusion_matrix
- Returns: dictionary of metrics

#### Function: `plot_confusion_matrix(conf_matrix, class_names, save_path=None, title=None)` (Lines 60-96)
```python
def plot_confusion_matrix(conf_matrix, class_names, save_path=None, title=None):
    """Plot and save confusion matrix heatmap"""
```
- Uses seaborn heatmap
- Adds thick border, bold labels
- Saves to PNG at 300 DPI

#### Function: `print_metrics(metrics)` (Lines 99-110)
```python
def print_metrics(metrics):
    """Print metrics to console"""
```
- Prints accuracy, precision, recall, F1, confusion matrix

---

### 2.7 `testing/run_test.py` (Lines 1-64)
**Purpose**: Batch testing on all trained runs

**No function definitions** - Script-level execution only

**Key Operations**:
- Loads `all_training_results.csv`
- Identifies completed test runs (skips duplicates)
- For each training run:
  - Updates config with best fold checkpoint
  - Sets `evaluation_flag=True`, `crossval_flag=False`
  - Calls `run.py` via subprocess

---

### 2.8 `cnn/models.py` (Lines 1-100)
**Purpose**: Neural network architectures

#### Class: `ResNet3D(nn.Module)` (Lines 7-17)
```python
class ResNet3D(nn.Module):
    def __init__(self, n_classes, in_channels=1):
        """3D ResNet18 adapted for FC maps"""
```
- Uses `torchvision.models.video.r3d_18`
- Modifies stem conv layer for single channel input
- Replaces final FC layer for `n_classes` outputs

#### Class: `DenseNet3D(nn.Module)` (Lines 21-31)
```python
class DenseNet3D(nn.Module):
    def __init__(self, n_classes):
        """3D DenseNet121 from MONAI"""
```
- Uses `monai.networks.nets.DenseNet121`
- Configured for 3D volumes, single input channel

#### Class: `VGG16_3D(nn.Module)` (Lines 34-100)
```python
class VGG16_3D(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        """Custom 3D VGG16 implementation"""
```
- 5 convolutional blocks with MaxPool3d
- 3 fully-connected layers (4096 → 4096 → num_classes)
- Dropout for regularization

---

### 2.9 `cnn/datasets.py` (Lines 1-129)
**Purpose**: PyTorch Dataset classes for loading FC maps

#### Class: `FCDataset(Dataset)` (Lines 7-54)
```python
class FCDataset(Dataset):
    def __init__(self, data_dir, df_labels, label_column, task, transform=None):
        """Standard (non-augmented) FC dataset"""
```
- Loads one `.processed.npy` file per subject
- Auto-maps string labels to indices
- Returns dict: `{'X': tensor, 'y': label, 'id': subject_id}`

#### Class: `AugmentedFCDataset(Dataset)` (Lines 57-129)
```python
class AugmentedFCDataset(Dataset):
    def __init__(self, data_dir, df_labels, label_column, task, transform=None):
        """Augmented FC dataset with multiple samples per subject"""
```
- Each subject has a folder with multiple `.npy` files
- Each file treated as separate sample
- Increases training set size via data augmentation

---

### 2.10 `utils/cnn_utils.py` (Lines 1-67)
**Purpose**: Helper functions for summaries

#### Function: `create_training_summary(params, best_fold_info, fold_accuracies, fold_val_losses, fold_train_losses)` (Lines 4-22)
```python
def create_training_summary(params, best_fold_info, fold_accuracies, fold_val_losses, fold_train_losses):
    """Create training summary row for CSV"""
```
- Returns dict with: run_id, group, seed, best fold, best accuracy, avg metrics, hyperparameters

#### Function: `create_tuning_summary(config_id, params, metrics)` (Lines 25-42)
```python
def create_tuning_summary(config_id, params, metrics):
    """Create tuning summary row for CSV"""
```
- Returns dict with: config_id, group, best fold, accuracies, hyperparameters

#### Function: `create_testing_summary(params, metrics)` (Lines 45-54)
```python
def create_testing_summary(params, metrics):
    """Create testing summary row for CSV"""
```
- Returns dict with: run_id, group, seed, accuracy, precision, recall, F1

#### Function: `resolve_split_csv_path(split_dir, group1, group2)` (Lines 57-67)
```python
def resolve_split_csv_path(split_dir, group1, group2):
    """Find correct split CSV (handles group order)"""
```
- Tries both `group1_group2` and `group2_group1` filenames
- Raises error if neither exists

---

## 3. Function Call Graph

### 3.1 Entry Points
1. **`run_train.py`** (main script)
   - Calls: `subprocess.run(["python", run_script_path])` → launches `run.py`
   
2. **`hyper_tuning.py`** (main script)
   - Calls: `tuning()` → `main_worker()` (from `run.py`)
   
3. **`run_test_tuning.py`** (main script)
   - Calls: `subprocess.run(["python", run_script_path])` → launches `run.py`
   
4. **`run_test.py`** (main script)
   - Calls: `subprocess.run(["python", run_script_path])` → launches `run.py`

5. **`run.py`** (if `__name__ == '__main__'`)
   - Calls: `main_worker(args)`

---

### 3.2 Core Call Chain (Training Mode)

```
run_train.py (loops over seeds)
    └─> subprocess → run.py::main_worker()
            ├─> set_seed()
            ├─> resolve_split_csv_path() [from cnn_utils]
            ├─> StratifiedKFold.split()
            └─> FOR EACH FOLD:
                    ├─> FCDataset() [from datasets]
                    ├─> AugmentedFCDataset() [from datasets]
                    ├─> ResNet3D() / DenseNet3D() / VGG16_3D() [from models]
                    ├─> run_epochs()
                    │       └─> FOR EACH EPOCH:
                    │               ├─> train() [from train.py]
                    │               ├─> validate() [from train.py]
                    │               └─> torch.save() [checkpoint]
                    │       └─> plot_losses() [from train.py]
                    └─> create_training_summary() [from cnn_utils]
```

---

### 3.3 Core Call Chain (Hyperparameter Tuning)

```
hyper_tuning.py::tuning()
    ├─> itertools.product() [generate grid combos]
    └─> FOR EACH VALID COMBO:
            ├─> is_valid_combo()
            ├─> main_worker(params with tuning_flag=True)
            │       └─> [same as training mode above]
            │       └─> RETURNS: best_fold_info dict
            └─> create_tuning_summary() [from cnn_utils]
```

---

### 3.4 Core Call Chain (Evaluation Mode)

```
run_test.py / run_test_tuning.py
    └─> subprocess → run.py::main_worker()
            ├─> set_seed()
            ├─> resolve_split_csv_path() [from cnn_utils]
            ├─> FCDataset() [test set]
            ├─> ResNet3D() / DenseNet3D() / VGG16_3D()
            ├─> torch.load() [checkpoint]
            ├─> evaluate() [from test.py]
            ├─> compute_metrics() [from test.py]
            ├─> plot_confusion_matrix() [from test.py]
            └─> create_testing_summary() [from cnn_utils]
```

---

## 4. Configuration Files

### 4.1 `cnn_config.json`
**Structure**:
```json
{
  "paths": {
    "data_dir": "path/to/processed/FCmaps",
    "data_dir_augmented": "path/to/augmented/FCmaps",
    "runs_dir": "path/to/save/results",
    "split_dir": "path/to/train_val_test_splits",
    "tuning_results_dir": "path/to/tuning/results"
  },
  "training": {
    "model_type": "resnet | densenet | vgg16",
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam | sgd",
    "n_folds": 5,
    "seed": 42
  },
  "fixed": {
    "label_column": "Group",
    "threshold": "no thr",
    "test_size": 0.2,
    "plot": true,
    "training_csv": true,
    "tuning_flag": false
  },
  "experiment": {
    "group1": "PSP",
    "group2": "CBS",
    "run_id": 1,
    "crossval_flag": true,
    "evaluation_flag": false,
    "ckpt_path_evaluation": null,
    "seed": 42
  }
}
```

**Key Parameters**:
- `crossval_flag=True`: Train with 5-fold CV on train set
- `evaluation_flag=True`: Evaluate on test set using checkpoint
- `tuning_flag=True`: Return metrics dict (used in grid search)
- `training_csv=True`: Save per-epoch results to Excel
- `plot=True`: Save learning curves and confusion matrix

---

### 4.2 `cnn_grid.json`
**Structure**:
```json
{
  "grid": {
    "optimizer": ["adam"],
    "batch_size": [4, 8, 16],
    "lr": [1e-3, 1e-4],
    "weight_decay": [0.001],
    "model_type": ["resnet", "densenet", "vgg16"],
    "epochs": [50],
    "n_folds": [5],
    "seed": [42]
  },
  "experiment": {
    "group1": "PSP",
    "group2": "CBS",
    "run_id": 8,
    "tuning_flag": true
  }
}
```

**Total Combinations**: 3 models × 3 batch_sizes × 2 lr = 18 configs
**Validation**: Filtered by `is_valid_combo()` for GPU memory constraints

---

## 5. Data Flow

### 5.1 Training Data Flow
```
Split CSV (train/val/test labels)
    ↓
FCDataset / AugmentedFCDataset
    ↓ (loads .npy files)
DataLoader (batches)
    ↓
Model (ResNet3D / DenseNet3D / VGG16_3D)
    ↓
train() / validate() functions
    ↓
Best checkpoint saved (torch.save)
    ↓
Training summary → CSV
```

### 5.2 Testing Data Flow
```
Checkpoint (.pt file)
    ↓
Load model.state_dict()
    ↓
FCDataset (test set)
    ↓
evaluate() function
    ↓
compute_metrics() → accuracy, precision, recall, F1
    ↓
Testing summary → CSV
```

---

## 6. Key Differences from ML Pipeline

### 6.1 Data Handling
- **ML**: Uses CSV with pre-computed features (UMAP, networks)
- **DL**: Loads raw 3D FC maps (91×109×91 volumes) from `.npy` files

### 6.2 Training Strategy
- **ML**: Single train/test split, seeds only affect model initialization
- **DL**: 5-fold CV on train set, best fold checkpoint used for test

### 6.3 Hyperparameter Tuning
- **ML**: GridSearchCV with inner CV on train set
- **DL**: Manual grid search, each config trains 5 folds, best fold selected

### 6.4 Augmentation
- **ML**: No data augmentation
- **DL**: `AugmentedFCDataset` provides multiple samples per subject

### 6.5 Output Files
- **ML**: Single CSV with all seeds' results
- **DL**: 
  - `all_training_results.csv`: CV results for each run
  - `all_testing_results.csv`: Test set results for each run
  - Per-run folders: checkpoints, plots, Excel logs

---

## 7. Execution Workflows

### 7.1 Standard Training (Multiple Seeds)
```bash
# Edit cnn_config.json:
# - crossval_flag: true
# - evaluation_flag: false
# - tuning_flag: false

python run_train.py

# Output: 5 runs (one per seed) in results/runs/run{1-5}/
# Summary: results/runs/all_training_results.csv
```

### 7.2 Hyperparameter Tuning
```bash
# Edit cnn_grid.json with grid parameters

python hyper_tuning.py

# Output: results/tuning/tuning{run_id}/config{1-18}/
# Summary: results/tuning/tuning{run_id}/grid_results.csv
```

### 7.3 Testing Best Models
```bash
# Option A: Test all trained runs
python run_test.py

# Option B: Test best tuning configs
python run_test_tuning.py

# Output: results/runs/all_testing_results.csv
```

---

## 8. Critical Code Patterns

### 8.1 Best Model Selection (Training)
```python
# In run_epochs():
if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_val_loss):
    best_accuracy = val_accuracy
    best_val_loss = val_loss
    torch.save(checkpoint, params['ckpt_path_evaluation'])
```

### 8.2 Best Fold Selection (After CV)
```python
# In main_worker():
for info in fold_infos:
    if (info['accuracy'] > best_fold_info['accuracy']
        or (info['accuracy'] == best_fold_info['accuracy'] and info['val_loss'] < best_fold_info['val_loss'])):
        best_fold_info = info
```

### 8.3 Checkpoint Loading (Testing)
```python
checkpoint = torch.load(params['ckpt_path_evaluation'], map_location=device, weights_only=True)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
```

---

## 9. Configuration Flags Summary

| Flag | Effect |
|------|--------|
| `crossval_flag=True` | Train with 5-fold CV on train set |
| `evaluation_flag=True` | Evaluate on test set using checkpoint |
| `tuning_flag=True` | Return metrics dict (no CSV saving) |
| `training_csv=True` | Save per-epoch metrics to Excel |
| `plot=True` | Save learning curves and confusion matrix |

**Valid Combinations**:
- Training only: `crossval=True, evaluation=False, tuning=False`
- Testing only: `crossval=False, evaluation=True, tuning=False`
- Training+Testing: `crossval=True, evaluation=True, tuning=False`
- Tuning: `crossval=True, evaluation=False, tuning=True`

**Invalid**:
- `crossval=True, evaluation=True, tuning=True` (raises error)

---

## 10. Reproducibility

### Seeds Control:
1. **Python random**: `random.seed(seed)`
2. **NumPy**: `np.random.seed(seed)`
3. **PyTorch**: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
4. **CUDNN**: `torch.backends.cudnn.deterministic=True`, `benchmark=False`

### StratifiedKFold:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['seed'])
```
- Same seed → same train/val splits across folds

### Result Stability:
- Same seed + same config → identical fold splits → identical results
- Different seeds → different initializations → performance variance

---

## 11. Memory Optimization

### Batch Size Constraints (GPU):
```python
def is_valid_combo(params):
    model = params["model_type"]
    batch = params["batch_size"]
    
    if model == "vgg16" and batch not in [4, 8, 16]:
        return False
    if model in ["resnet", "densenet"] and batch not in [4, 16]:
        return False
    return True
```
- VGG16: Larger memory footprint → smaller batch sizes
- ResNet/DenseNet: More efficient → larger batch sizes allowed

---

## 12. Output Files Structure

```
results/
├── runs/
│   ├── all_training_results.csv   # CV summaries for all runs
│   ├── all_testing_results.csv    # Test summaries for all runs
│   ├── run1/
│   │   ├── best_model_fold1.pt    # Checkpoint for fold 1
│   │   ├── best_model_fold5.pt    # Checkpoint for fold 5
│   │   ├── log_train1             # Training logs
│   │   ├── training_folds.xlsx    # Per-epoch metrics
│   │   └── plots/                 # Learning curves
│   └── run5/
│       └── ...
└── tuning/
    └── tuning8/
        ├── grid_results.csv       # All configs' results
        ├── config1/
        │   ├── best_model_fold1.pt
        │   └── log_train_run8_config1
        └── config18/
            └── ...
```

---

## Summary

This DL pipeline implements a **fixed train/val/test split** approach with:
- **5-fold stratified CV** on the training set to select best fold
- **Best checkpoint** from CV used to evaluate on held-out test set
- **Multiple seeds** to test initialization stability
- **Grid search** for hyperparameter optimization
- **Data augmentation** via multiple HCP-bootstrapped samples

The key distinction from the ML pipeline is that **DL uses CV to select the best fold/checkpoint**, then evaluates that single model on the test set, whereas **ML uses CV only for hyperparameter selection** with GridSearchCV, then retrains on full train set.
