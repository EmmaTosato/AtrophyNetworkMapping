# Deep Learning Function Reference

## Complete Function Catalog with Line Numbers and Signatures

---

## 1. Training Module

### 1.1 `training/run.py`

| Function | Lines | Signature | Returns | Purpose |
|----------|-------|-----------|---------|---------|
| `set_seed` | 28-38 | `(seed: int)` | `None` | Set all random seeds for reproducibility |
| `run_epochs` | 41-121 | `(model, train_loader, val_loader, criterion, optimizer, params, fold)` | `Tuple[float, float, float, int]` | Train for N epochs, return best metrics |
| `main_worker` | 124-397 | `(params: dict, config_id: int = None)` | `Optional[dict]` | Main training/evaluation pipeline |

**Key Details**:
- `run_epochs` returns: `(best_accuracy, best_train_loss, best_val_loss, best_epoch)`
- `main_worker` returns: `None` (normal mode) or `dict` (tuning mode)
- `main_worker` modes:
  - `crossval_flag=True`: Training with 5-fold CV
  - `evaluation_flag=True`: Testing on test set
  - `tuning_flag=True`: Return metrics dict (no CSV saving)

### 1.2 `training/train.py`

| Function | Lines | Signature | Returns | Purpose |
|----------|-------|-----------|---------|---------|
| `train` | 5-36 | `(model, train_loader, criterion, optimizer, device)` | `float` | One epoch of training |
| `validate` | 38-71 | `(model, val_loader, criterion, device)` | `Tuple[float, float]` | Evaluate on validation set |
| `plot_losses` | 73-100 | `(train_losses, val_losses, val_accuracies=None, save_path=None, title=None)` | `None` | Plot and save learning curves |

**Key Details**:
- `train` returns: average training loss
- `validate` returns: `(val_loss, val_accuracy)`
- `plot_losses` saves to PNG file

### 1.3 `training/hyper_tuning.py`

| Function | Lines | Signature | Returns | Purpose |
|----------|-------|-----------|---------|---------|
| `is_valid_combo` | 8-21 | `(params: dict)` | `bool` | Check if hyperparameter combo is GPU-feasible |
| `tuning` | 23-103 | `(base_args_path: str, grid_path: str)` | `None` | Perform grid search over hyperparameters |

**Key Details**:
- `is_valid_combo` enforces:
  - VGG16: batch_size in [4, 8, 16]
  - ResNet/DenseNet: batch_size in [4, 16]
- `tuning` saves results to `grid_results.csv`

---

## 2. Testing Module

### 2.1 `testing/test.py`

| Function | Lines | Signature | Returns | Purpose |
|----------|-------|-----------|---------|---------|
| `evaluate` | 7-32 | `(model, loader, device)` | `Tuple[np.ndarray, np.ndarray]` | Get predictions on test set |
| `compute_metrics` | 35-57 | `(y_true, y_pred)` | `dict` | Calculate classification metrics |
| `plot_confusion_matrix` | 60-96 | `(conf_matrix, class_names, save_path=None, title=None)` | `None` | Plot confusion matrix heatmap |
| `print_metrics` | 99-110 | `(metrics: dict)` | `None` | Print metrics to console |

**Key Details**:
- `evaluate` returns: `(y_true, y_pred)` as NumPy arrays
- `compute_metrics` returns: `{'accuracy': float, 'precision': float, 'recall': float, 'f1': float, 'confusion_matrix': np.ndarray}`
- `plot_confusion_matrix` saves to PNG at 300 DPI

---

## 3. CNN Module

### 3.1 `cnn/models.py`

| Class | Lines | Constructor | Forward Input | Forward Output | Architecture |
|-------|-------|-------------|---------------|----------------|--------------|
| `ResNet3D` | 7-17 | `(n_classes, in_channels=1)` | `(B, 1, 91, 109, 91)` | `(B, n_classes)` | torchvision r3d_18 |
| `DenseNet3D` | 21-31 | `(n_classes)` | `(B, 1, 91, 109, 91)` | `(B, n_classes)` | MONAI DenseNet121 |
| `VGG16_3D` | 34-100 | `(num_classes=2, input_channels=1)` | `(B, 1, 91, 109, 91)` | `(B, num_classes)` | Custom VGG16 3D |

**Key Details**:
- All models adapted for single-channel 3D input
- ResNet3D: Modified stem conv layer (3×7×7, stride 1×2×2)
- DenseNet3D: MONAI pretrained=False
- VGG16_3D: 5 conv blocks + 3 FC layers (4096→4096→num_classes)

### 3.2 `cnn/datasets.py`

| Class | Lines | Constructor | `__len__` | `__getitem__` Returns | Data Source |
|-------|-------|-------------|-----------|------------------------|-------------|
| `FCDataset` | 7-54 | `(data_dir, df_labels, label_column, task, transform=None)` | `len(samples)` | `{'X': tensor, 'y': label, 'id': str}` | Single .npy per subject |
| `AugmentedFCDataset` | 57-129 | `(data_dir, df_labels, label_column, task, transform=None)` | `len(samples)` | `{'X': tensor, 'y': label, 'id': str}` | Multiple .npy per subject |

**Key Details**:
- `FCDataset`: Loads `{subject_id}.processed.npy`
- `AugmentedFCDataset`: Loads all `.npy` in `{subject_id}/` folder
- Both auto-map string labels to indices if needed
- Shape: `(1, 91, 109, 91)` tensor

---

## 4. Utils Module

### 4.1 `utils/cnn_utils.py`

| Function | Lines | Signature | Returns | Purpose |
|----------|-------|-----------|---------|---------|
| `create_training_summary` | 4-22 | `(params, best_fold_info, fold_accuracies, fold_val_losses, fold_train_losses)` | `dict` | Create training summary row for CSV |
| `create_tuning_summary` | 25-42 | `(config_id, params, metrics)` | `dict` | Create tuning summary row for CSV |
| `create_testing_summary` | 45-54 | `(params, metrics)` | `dict` | Create testing summary row for CSV |
| `resolve_split_csv_path` | 57-67 | `(split_dir, group1, group2)` | `str` | Find correct split CSV file |

**Key Details**:
- All summary functions return dict with rounded values (3 decimals)
- `resolve_split_csv_path` tries both `group1_group2` and `group2_group1` filenames

---

## 5. Entry Point Scripts

### 5.1 `training/run_train.py`
**Type**: Script (no functions)
**Lines**: 1-59
**Purpose**: Launch multiple training runs with different seeds

**Logic**:
```python
for seed in [42, 123, 2023, 31415, 98765]:
    update_config(seed=seed, run_id=next_run_id)
    subprocess.run(["python", "run.py"])
    next_run_id += 1
```

### 5.2 `training/run_test_tuning.py`
**Type**: Script (no functions)
**Lines**: 1-96
**Purpose**: Test best configurations from hyperparameter tuning

**Logic**:
```python
df = pd.read_csv(tuning_csv)
for _, row in df.iterrows():
    update_config(best_hyperparams=row)
    subprocess.run(["python", "run.py"])
```

### 5.3 `testing/run_test.py`
**Type**: Script (no functions)
**Lines**: 1-64
**Purpose**: Batch testing on all trained runs

**Logic**:
```python
df = pd.read_csv(training_results_csv)
for _, row in df.iterrows():
    if run_id not in completed_runs:
        update_config(checkpoint=best_fold_ckpt)
        subprocess.run(["python", "run.py"])
```

---

## 6. Function Call Dependencies

### Training Flow (Simplified)
```
run_train.py
    └─> subprocess → run.py::main_worker()
            ├─> set_seed()
            ├─> resolve_split_csv_path()
            ├─> FCDataset / AugmentedFCDataset (instantiate)
            ├─> ResNet3D / DenseNet3D / VGG16_3D (instantiate)
            └─> run_epochs()
                    ├─> train()
                    ├─> validate()
                    ├─> torch.save()
                    └─> plot_losses()
            └─> create_training_summary()
```

### Testing Flow (Simplified)
```
run_test.py
    └─> subprocess → run.py::main_worker()
            ├─> set_seed()
            ├─> resolve_split_csv_path()
            ├─> FCDataset (instantiate)
            ├─> ResNet3D / DenseNet3D / VGG16_3D (instantiate)
            ├─> torch.load()
            ├─> evaluate()
            ├─> compute_metrics()
            ├─> plot_confusion_matrix()
            └─> create_testing_summary()
```

### Tuning Flow (Simplified)
```
hyper_tuning.py::tuning()
    ├─> itertools.product() (generate grid)
    └─> FOR EACH valid combo:
            ├─> is_valid_combo()
            ├─> main_worker(tuning_flag=True)
            │       └─> [same as training flow]
            │       └─> RETURNS metrics dict
            └─> create_tuning_summary()
```

---

## 7. External Library Dependencies

### PyTorch Functions
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.save(checkpoint, path)`
- `torch.load(path, map_location=device)`
- `torch.nn.CrossEntropyLoss()`
- `torch.optim.Adam()`, `torch.optim.SGD()`
- `torch.utils.data.DataLoader()`

### Scikit-learn Functions
- `sklearn.model_selection.StratifiedKFold()`
- `sklearn.metrics.accuracy_score()`
- `sklearn.metrics.precision_score()`
- `sklearn.metrics.recall_score()`
- `sklearn.metrics.f1_score()`
- `sklearn.metrics.confusion_matrix()`

### Other Libraries
- `pandas.read_csv()`, `pandas.DataFrame.to_csv()`
- `numpy.load()`, `numpy.mean()`
- `matplotlib.pyplot.plot()`, `seaborn.heatmap()`
- `subprocess.run(["python", script_path])`
- `itertools.product(*grid.values())`

---

## 8. Configuration Parameters Reference

### Training Parameters (cnn_config.json)
```json
{
  "model_type": "resnet | densenet | vgg16",
  "epochs": 50,
  "batch_size": 4 | 8 | 16,
  "lr": 0.001 | 0.0001,
  "weight_decay": 0.0001 | 0.001,
  "optimizer": "adam | sgd",
  "n_folds": 5,
  "seed": 42
}
```

### Experiment Flags (cnn_config.json)
```json
{
  "crossval_flag": true,      // Train with 5-fold CV
  "evaluation_flag": false,   // Evaluate on test set
  "tuning_flag": false,       // Return metrics dict (grid search)
  "plot": true,               // Save plots
  "training_csv": true        // Save per-epoch Excel
}
```

### Valid Flag Combinations
| `crossval` | `evaluation` | `tuning` | Mode | Description |
|------------|--------------|----------|------|-------------|
| ✅ True | ❌ False | ❌ False | Training | Train 5-fold CV, save checkpoints |
| ❌ False | ✅ True | ❌ False | Testing | Load checkpoint, test on test set |
| ✅ True | ✅ True | ❌ False | Both | Train + Test in one run |
| ✅ True | ❌ False | ✅ True | Tuning | Return metrics, no CSV |
| ✅ True | ✅ True | ✅ True | ❌ INVALID | Raises error |

---

## 9. Model Architecture Details

### ResNet3D
- **Base**: `torchvision.models.video.r3d_18`
- **Modification**: Stem conv layer `Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2))`
- **Final layer**: `Linear(512, n_classes)`
- **Parameters**: ~33M

### DenseNet3D
- **Base**: `monai.networks.nets.DenseNet121`
- **Configuration**: `spatial_dims=3, in_channels=1, out_channels=n_classes`
- **Parameters**: ~7M

### VGG16_3D
- **Architecture**: Custom implementation
- **Blocks**: 5 conv blocks (64→128→256→512→512)
- **Classifier**: FC(6144→4096→4096→n_classes)
- **Parameters**: ~138M

---

## 10. Dataset Statistics

### Original Dataset
- **Train set**: 104 subjects (before augmentation)
- **Test set**: 26 subjects (fixed)
- **Classes**: 2 (e.g., PSP vs CBS)

### Augmented Dataset (Training Only)
- **Samples per subject**: ~50 (via HCP bootstrapping)
- **Total train samples**: ~5200 (104 × 50)
- **Test samples**: 26 (no augmentation)

### Data Format
- **Input shape**: (1, 91, 109, 91)
- **Voxel size**: 2mm isotropic
- **Data type**: float32
- **File format**: `.npy` (NumPy array)

---

## Summary Statistics

| Category | Count | Notes |
|----------|-------|-------|
| **Files** | 10 | 5 training, 2 testing, 2 CNN, 1 utils |
| **Functions** | 15 | Excluding class methods |
| **Classes** | 5 | 3 models, 2 datasets |
| **Entry Scripts** | 3 | run_train, run_test, run_test_tuning |
| **Config Files** | 2 | cnn_config.json, cnn_grid.json |
| **Total Lines of Code** | ~1500 | Excluding comments |

---

**Last updated**: 2025-01-19
