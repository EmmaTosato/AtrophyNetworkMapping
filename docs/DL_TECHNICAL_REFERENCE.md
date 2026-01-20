# 🔬 Deep Learning Pipeline - Technical Documentation

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Data Flow](#data-flow)
4. [Function Reference](#function-reference)
5. [Configuration System](#configuration-system)
6. [Training Mechanics](#training-mechanics)
7. [Testing & Evaluation](#testing--evaluation)

---

## 1. Architecture Overview

### System Components

```
src/DL_analysis/
├── cnn/
│   ├── models.py          # Neural network architectures
│   └── datasets.py        # Data loading classes
├── training/
│   ├── run.py             # Main orchestrator
│   ├── train.py           # Training/validation loops
│   ├── hyper_tuning.py    # Grid search
│   ├── run_train.py       # Multi-seed runner
│   └── run_test_tuning.py # Test tuning configs
├── testing/
│   ├── test.py            # Evaluation functions
│   └── run_test.py        # Test runner for runs
├── config/
│   ├── cnn_config.json    # Base configuration
│   └── cnn_grid.json      # Hyperparameter grid
└── utils/
    └── cnn_utils.py       # Helper functions
```

---

## 2. Code Structure

### 2.1 Main Entry Point: `run.py`

**Function**: `main_worker(params, config_id=None)`
- **Purpose**: Orchestrates training and/or evaluation
- **Modes**:
  1. `crossval_flag=True, evaluation_flag=False`: Training with CV
  2. `crossval_flag=False, evaluation_flag=True`: Testing only
  3. `crossval_flag=True, evaluation_flag=True`: Full pipeline
  4. `tuning_flag=True`: Grid search mode (returns dict, no CSV)

**Key Logic**:
```python
if crossval_flag:
    for fold in skf.split():
        train_on_fold()
        select_best_fold()
    save_checkpoints()

if evaluation_flag:
    load_best_checkpoint()
    test_on_test_set()
    save_metrics()
```

### 2.2 Training Loop: `train.py`

**Function**: `run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold)`
- **Returns**: `best_accuracy, best_train_loss, best_val_loss, best_epoch`
- **Early Stopping**: 
  - Primary: `val_accuracy` (maximize)
  - Tiebreaker: `val_loss` (minimize)
- **Checkpoint**: Saves at each improvement

**Function**: `train(model, train_loader, criterion, optimizer, device)`
- **Returns**: `avg_train_loss`
- **Details**: Single epoch training

**Function**: `validate(model, val_loader, criterion, device)`
- **Returns**: `avg_val_loss, val_accuracy`
- **Details**: Evaluation on validation set

### 2.3 Models: `models.py`

#### ResNet3D
```python
class ResNet3D(nn.Module):
    def __init__(self, n_classes=2):
        # Uses torchvision.models.video.r3d_18
        # Parameters: ~33M
        # Input: (B, 1, 91, 109, 91)
        # Output: (B, n_classes)
```

#### DenseNet3D
```python
class DenseNet3D(nn.Module):
    def __init__(self, n_classes=2):
        # Uses MONAI DenseNet121
        # Parameters: ~7M
        # Input: (B, 1, 91, 109, 91)
        # Output: (B, n_classes)
```

#### VGG16_3D
```python
class VGG16_3D(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        # Custom implementation
        # Parameters: ~138M
        # Input: (B, 1, 91, 109, 91)
        # Output: (B, num_classes)
```

### 2.4 Datasets: `datasets.py`

#### FCDataset
```python
class FCDataset(Dataset):
    """
    Standard dataset loader.
    Each subject: 1 .npy file
    Used for: Validation and Test sets
    """
    def __getitem__(self, idx):
        # Load: subject_id.processed.npy
        # Return: (1, 91, 109, 91), label
```

#### AugmentedFCDataset
```python
class AugmentedFCDataset(Dataset):
    """
    Augmented dataset loader.
    Each subject: N .npy files (50× augmentation)
    Used for: Training sets only
    """
    def __getitem__(self, idx):
        # Load: subject_id.aug{N}.processed.npy
        # Return: (1, 91, 109, 91), label
```

---

## 3. Data Flow

### 3.1 Training Flow

```
assets/split_cnn/ADNI_PSP_splitted.csv
              ▼
     df[split='train']  →  105 subjects
              ▼
   StratifiedKFold(n_splits=5, random_state=seed)
              ▼
   ┌──────────────────────┬──────────────────────┐
   │   Fold 1-5 Split     │   Validation Split   │
   │   ~84 subjects       │   ~21 subjects       │
   └──────────────────────┴──────────────────────┘
              ▼                      ▼
    AugmentedFCDataset           FCDataset
    (50× augmented)           (original only)
              ▼                      ▼
       Train Loader              Val Loader
              ▼                      ▼
         train()                 validate()
              ▼                      ▼
       Update weights      Compute val_accuracy
              ▼
    Early Stopping → Save best_model_fold{N}.pt
```

### 3.2 Testing Flow

```
all_training_results.csv
              ▼
   Extract: run_id, best_fold, hyperparameters
              ▼
   Load: best_model_fold{best_fold}.pt
              ▼
   df[split='test'] → 27 subjects
              ▼
         FCDataset
              ▼
        Test Loader
              ▼
        evaluate()
              ▼
   compute_metrics() → accuracy, precision, recall, f1
              ▼
   Save: all_testing_results.csv
```

---

## 4. Function Reference

### 4.1 Core Functions

| Function | File | Purpose | Returns |
|----------|------|---------|---------|
| `main_worker()` | run.py | Main orchestrator | dict (tuning) / None |
| `run_epochs()` | train.py | Train one fold | best_acc, loss, epoch |
| `train()` | train.py | Single epoch train | avg_train_loss |
| `validate()` | train.py | Single epoch val | avg_val_loss, accuracy |
| `evaluate()` | test.py | Test evaluation | y_true, y_pred |
| `compute_metrics()` | test.py | Calculate metrics | dict (acc, prec, rec, f1) |
| `tuning()` | hyper_tuning.py | Grid search | None (saves CSV) |

### 4.2 Utility Functions

| Function | File | Purpose |
|----------|------|---------|
| `set_seed()` | run.py | Set reproducibility |
| `create_training_summary()` | cnn_utils.py | Format training results |
| `create_testing_summary()` | cnn_utils.py | Format testing results |
| `create_tuning_summary()` | cnn_utils.py | Format tuning results |
| `resolve_split_csv_path()` | cnn_utils.py | Build split CSV path |
| `plot_losses()` | train.py | Save learning curves |
| `plot_confusion_matrix()` | test.py | Save confusion matrix |

---

## 5. Configuration System

### 5.1 cnn_config.json Structure

```json
{
  "paths": {
    "data_dir_augmented": "FCmaps_augmented_processed/",
    "data_dir": "FCmaps_processed/",
    "runs_dir": "results/runs/",
    "split_dir": "assets/split_cnn/",
    "tuning_results_dir": "results/tuning/"
  },
  "training": {
    "model_type": "densenet",
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
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
    "group1": "ADNI",
    "group2": "PSP",
    "run_id": 1,
    "crossval_flag": true,
    "evaluation_flag": false,
    "ckpt_path_evaluation": null,
    "seed": 42
  }
}
```

### 5.2 cnn_grid.json Structure

```json
{
  "grid": {
    "optimizer": ["adam", "sgd"],
    "batch_size": [4, 8, 16],
    "lr": [1e-3, 1e-4, 1e-5],
    "weight_decay": [1e-4, 1e-3],
    "model_type": ["resnet", "densenet", "vgg16"],
    "epochs": [50],
    "n_folds": [5],
    "seed": [42]
  },
  "experiment": {
    "group1": "ADNI",
    "group2": "PSP",
    "run_id": 1,
    "tuning_flag": true
  }
}
```

### 5.3 Configuration Merging

```python
args = {
    **config["paths"],
    **config["training"],
    **config["fixed"],
    **config["experiment"]
}
```

---

## 6. Training Mechanics

### 6.1 Seed Management

**Two Types of Seeds**:
1. **StratifiedKFold seed**: Controls fold splits
   ```python
   skf = StratifiedKFold(n_splits=5, random_state=params['seed'])
   ```
2. **PyTorch seed**: Controls weight initialization
   ```python
   set_seed(params['seed'])
   torch.manual_seed(seed)
   ```

⚠️ **Critical**: In current implementation, **same seed** controls both!
- Different seeds → Different fold assignments → Not comparable
- Solution: Fix KFold seed (42), vary PyTorch seed only

### 6.2 Cross-Validation Loop

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
    # 1. Subset dataframes
    fold_train_df = train_df.iloc[train_idx]
    fold_val_df = train_df.iloc[val_idx]
    
    # 2. Create datasets
    train_dataset = AugmentedFCDataset(...)  # 50× augmented
    val_dataset = FCDataset(...)             # Original only
    
    # 3. Train
    best_acc, best_loss, best_epoch = run_epochs(...)
    
    # 4. Track best fold
    if best_acc > global_best_acc:
        global_best_fold = fold
        global_best_checkpoint = f"best_model_fold{fold}.pt"
```

### 6.3 Best Fold Selection

**Criteria** (in order):
1. **Highest validation accuracy**
2. **Lowest validation loss** (tiebreaker)

```python
if (info['accuracy'] > best_fold_info['accuracy'] or
    (info['accuracy'] == best_fold_info['accuracy'] and 
     info['val_loss'] < best_fold_info['val_loss'])):
    best_fold_info = info
```

### 6.4 Checkpoint Structure

```python
checkpoint = {
    'state_dict': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'val_accuracy': val_accuracy,
    'epoch': best_epoch,
    'best_train_loss': best_train_loss,
    'best_val_loss': best_val_loss,
    'fold': fold
}
```

---

## 7. Testing & Evaluation

### 7.1 Evaluation Function

```python
def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred)
```

### 7.2 Metrics Computation

```python
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
```

---

## 8. Output Files

### 8.1 Training Outputs

**all_training_results.csv**:
```
run_id,group,seed_x,threshold,best fold,best epoch,best accuracy,
best validation loss,average accuracy,model_type,optimizer,lr,
batch_size,weight_decay,epochs,test size
```

**training_folds.xlsx** (per run):
- Sheet per fold: Epoch, Train Loss, Val Loss, Val Accuracy

### 8.2 Testing Outputs

**all_testing_results.csv**:
```
run_id,group,seed,accuracy,precision,recall,f1
```

### 8.3 Tuning Outputs

**grid_results.csv** (per tuning):
```
config,group,threshold,best_fold,best_accuracy,avg_accuracy,
avg_train_loss,avg_val_loss,optimizer,lr,batch_size,weight_decay,
model_type,epochs,test size
```

**merged_{GROUP}.csv**:
- Aggregates tuning results with test metrics

---

## 9. Performance Notes

### 9.1 Memory Management

| Model | Batch Size 4 | Batch Size 8 | Batch Size 16 |
|-------|--------------|--------------|---------------|
| ResNet3D | ✅ 3GB | ✅ 6GB | ✅ 12GB |
| DenseNet3D | ✅ 2GB | ✅ 4GB | ✅ 8GB |
| VGG16_3D | ✅ 8GB | ✅ 16GB | ❌ OOM |

### 9.2 Training Time Estimates

**Per Fold** (50 epochs):
- ResNet3D: ~20 min
- DenseNet3D: ~15 min
- VGG16_3D: ~30 min

**Full Run** (5 folds):
- ResNet3D: ~1.5 hours
- DenseNet3D: ~1.2 hours
- VGG16_3D: ~2.5 hours

**Grid Search** (24 configs × 5 folds):
- Total: ~30-40 hours

---

## 10. Debugging & Logs

### 10.1 Log Files

| Log File | Content |
|----------|---------|
| `log_train{N}` | CV training output |
| `log_test{N}` | Testing output |
| `log_total{N}` | Full pipeline output |
| `log_train_run{M}_config{N}` | Tuning output |

### 10.2 Common Issues

**Issue**: OOM during training
- **Solution**: Reduce batch size or use smaller model

**Issue**: Different results with same seed
- **Cause**: cudnn non-determinism
- **Solution**: `torch.backends.cudnn.deterministic = True`

**Issue**: Low test accuracy
- **Expected**: Dataset is small and task is difficult
- **Not a bug**: Focus on variance across seeds

---

**Autore**: Technical deep-dive per AI agents e sviluppatori  
**Ultima modifica**: Gennaio 2026
