# NESTED CROSS-VALIDATION - DL/ML IMPLEMENTATION

**Date:** January 22, 2026
**Version:** 2.0 - Technical Documentation (As Built)
**Conda environment:** anm (`conda activate anm`)

---

## 1. NESTED CV ARCHITECTURE

```

OUTER CROSS-VALIDATION (5-fold StratifiedKFold)
 • Fixed Seed: 42 (reproducibility)
 • Split: 80% Train (~105 subj) / 20% Test (~27 subj)
 • For each fold: distinct train, distinct test

                            

 FOLD 1 OUTER
 Train: ~84 subjects | Test: ~21 subjects

                            

 INNER CROSS-VALIDATION (5-fold StratifiedKFold)
 • On Outer Train ONLY (~84 subj)
 • Grid Search: N hyperparameter configurations
 • Validation: inner fold (ORIGINAL data)
 • Seed: 42 (same as outer to ensure stability)

                            
         
          Inner Fold 1: Train ~67 | Val ~17
          Inner Fold 2: Train ~67 | Val ~17
          Inner Fold 3: Train ~67 | Val ~17
          Inner Fold 4: Train ~67 | Val ~17
          Inner Fold 5: Train ~67 | Val ~17
         
                            

 BEST HYPERPARAMETERS SELECTION
 • Mean accuracy over 5 inner folds
 • Configuration with highest accuracy → BEST PARAMS

                            

 FULL RETRAIN (with best params)
 • Train: ALL 84 subjects of the outer fold
 • Data AUGMENTED: 10× HCP bootstrap
 • Epochs: until convergence (with early stopping)

                            

 TESTING FOLD 1
 • Test: 21 subjects of the outer fold
 • ORIGINAL data (NO augmentation)
 • Metrics: Accuracy, F1, Precision, Recall, AUC

                            
       [REPEAT for FOLD 2, 3, 4, 5 OUTER]
                            

 FINAL RESULTS AGGREGATION
 • Mean ± Std over 5 outer folds
 • Per-fold metrics (full table)
 • Best model selection: fold with highest accuracy
 • Save: `results/DL/GROUP_GROUP/MODEL/aggregated_results.json`

```

---

## 2. DATA AUGMENTATION (10× HCP BOOTSTRAP)

**CORE POLICY:**
-  **AUGMENTATION**: ONLY in TRAINING sets (outer train, inner train during grid search, full retrain)
-  **NO AUGMENTATION**: ALWAYS in validation and test (original data)

**Dataset Classes:**
- `FCDataset`: original data (validation and test)
- `AugmentedFCDataset`: 10× augmented (training)

**Integration with Dynamic Splitting:**
- During Outer CV: `AugmentedFCDataset` loads from `data/FCmaps_augmented_processed/`
- Each subject has 10 `.npy` files (10 different HCP bootstraps)
- Validation/Test: `FCDataset` loads from `data/FCmaps_processed/` (originals)

---

## 3. GRID SEARCH AND INNER CV

**GRID SEARCH PLACEMENT:**
- Grid search: INSIDE outer loop
- Executed on: Outer Train (~84 subj)
- Inner CV: 5-fold for validation
- NO data leakage: outer test is NEVER seen during grid search

---

## 4. MODELS AND HYPERPARAMETERS (Literature)

### **AlexNet3D** (NEW - from Original Paper Krizhevsky 2012)

**Architecture:**
```python
Input: (1, 91, 109, 91)
Conv1: 96 filters, 11×11×11, stride=4 → ReLU → MaxPool 3×3×3 stride=2
Conv2: 256 filters, 5×5×5 → ReLU → MaxPool 3×3×3 stride=2
Conv3: 384 filters, 3×3×3 → ReLU
Conv4: 384 filters, 3×3×3 → ReLU
Conv5: 256 filters, 3×3×3 → ReLU → MaxPool 3×3×3 stride=2
Flatten → FC1: 4096 → Dropout(0.5) → ReLU
       → FC2: 4096 → Dropout(0.5) → ReLU
       → FC3: num_classes (2) → Softmax
```

**Original Hyperparameters (Paper):**
- Learning Rate: **0.01**
- Batch Size: **128**
- Momentum: **0.9**
- Weight Decay: **0.0005**
- Dropout: **0.5**
- Epochs: **90**
- Optimizer: **SGD + Momentum**

**Adaptations (Grid Search - Single Value):**
```python
{
    'lr': [0.01],
    'batch_size': [8],
    'weight_decay': [5e-4],
    'epochs': [40],
    'optimizer': ['sgd'],
    'momentum': [0.9],
    'scheduler_gamma': [0.1],
    'scheduler_patience': [5],
    'patience': [null]
}
```

---

### **ResNet18** (r3d_18 - from Paper He 2015)

**Architecture:**
```python
# torchvision.models.video.r3d_18
Input: (1, 91, 109, 91)
Conv1: 64 filters, 7×7×7, stride=2 → BN → ReLU → MaxPool 3×3×3
Layer1: [BasicBlock3D × 2] → 64 channels
Layer2: [BasicBlock3D × 2] → 128 channels (downsample)
Layer3: [BasicBlock3D × 2] → 256 channels (downsample)
Layer4: [BasicBlock3D × 2] → 512 channels (downsample)
AdaptiveAvgPool3D → FC: num_classes
```

**Original Hyperparameters (Paper):**
- Learning Rate: **0.1** (0.01 warmup)
- Batch Size: **256**
- Momentum: **0.9**
- Weight Decay: **0.0001**
- NO Dropout
- Epochs: **60**
- Optimizer: **SGD + Momentum**

**Adaptations (Grid Search - Single Value):**
```python
{
    'lr': [1e-2],                 # Fixed 0.01 (Safe warmup/batch adj)
    'batch_size': [8],             # Fixed (constraint)
    'weight_decay': [1e-4],        # Original
    'epochs': [40],                # Reduced
    'patience': [15]               # Early Stopping
}
```

---

### **VGG16_3D** (from Paper Simonyan 2014)

**Architecture:**
```python
# Configuration D (VGG16)
Input: (1, 91, 109, 91)
Block1: Conv64×2 → MaxPool
Block2: Conv128×2 → MaxPool
Block3: Conv256×3 → MaxPool
Block4: Conv512×3 → MaxPool
Block5: Conv512×3 → MaxPool
FC1: 4096 → Dropout(0.5) → ReLU
FC2: 4096 → Dropout(0.5) → ReLU
FC3: num_classes → Softmax
```

**Original Hyperparameters (Paper):**
- Learning Rate: **0.01**
- Batch Size: **256**
- Momentum: **0.9**
- Weight Decay: **5e-4**
- Dropout: **0.5**
- Epochs: **74**
- Optimizer: **SGD + Momentum**

**Adaptations (Grid Search - Single Value):**
```python
{
    'lr': [1e-2],                 # Fixed 0.01 (Original)
    'batch_size': [8],             # Fixed (constraint)
    'weight_decay': [5e-4],        # Original
    'epochs': [40],                # Reduced
    'patience': [15]               # Early Stopping
}
```

---

## 5. AGGREGATED RESULTS

**For each Outer Fold (5 total):**
```json
{
  "fold_1": {
    "best_hyperparams": {"lr": 0.01, "batch_size": 32, "weight_decay": 0.0005},
    "inner_cv_mean_acc": 0.78,
    "test_accuracy": 0.82,
    "test_f1": 0.81,
    "test_precision": 0.83,
    "test_recall": 0.80,
    "test_auc": 0.85,
    "confusion_matrix": [[10, 1], [2, 8]]
  },
  // ... fold_2 to fold_5
}
```

**Final Aggregation:**
```json
{
  "mean_accuracy": 0.80,
  "std_accuracy": 0.03,
  "mean_f1": 0.79,
  "std_f1": 0.04,
  "mean_auc": 0.84,
  "std_auc": 0.02,
  "best_fold": 1,
  "best_fold_accuracy": 0.82,
  "hyperparams_frequency": {
    "lr_0.01": 3,
    "lr_0.005": 2,
    "batch_32": 4,
    "batch_16": 1
  }
}
```

**Reported Metrics:**
- **Mean ± Std**: Accuracy, F1, Precision, Recall, AUC over 5 outer folds
- **Per-fold**: Full table with all metrics
- **Best Hyperparams**: Frequency of selection for each parameter
- **Best Model**: Fold with best performance (for deployment)

**Output Structure:**
```
results/DL/
 AD_PSP/
    resnet/
    fold_1/
       best_model.pt
       metrics.json
       confusion_matrix.png
       training_curves.png
    fold_2/ ... fold_5/
    aggregated_results.json
 ADNI_CBS/ (same structure)
 PSP_CBS/ (same structure)

---

## 6. IMPLEMENTATION (Python Scripts)

### **6.1 Execution Flow**
The system is **Config-Driven**.

1.  **Edit** `src/DL_analysis/config/cnn.json` (Job List)
2.  **Run** `src/DL_analysis/training/run_all.py`
    *   *Optional*: use `--grid_config` to change parameter file.

### **6.2 Core Runner: `run_nested_cv.py`**
    
    def nested_cv_classification(args):
    """
    Nested CV: Outer 5-fold + Inner 5-fold Grid Search
    Uses JSON configuration for parameters and paths.
    """
    # 1. Load metadata
    df_meta = load_metadata(group1, group2)
    
    # 2. Outer CV: StratifiedKFold(5, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n=== OUTER FOLD {outer_fold+1}/5 ===")
        
        # 3. Split outer
        train_subjects = df_meta.iloc[train_idx]
        test_subjects = df_meta.iloc[test_idx]
        
        # 4. Inner CV Grid Search (on train_subjects)
        best_params = inner_cv_grid_search(
            train_subjects, model_name, grid_params
        )
        
        # 5. Full Retrain (train_subjects, AUGMENTED, best_params)
        model = full_retrain(
            train_subjects, model_name, best_params,
            use_augmentation=True
        )
        
        # 6. Test (test_subjects, ORIGINAL)
        metrics = test_model(model, test_subjects, use_augmentation=False)
        results.append(metrics)
    
    # 7. Aggregate results
    aggregate_and_save(results, group1, group2, model_name)


def inner_cv_grid_search(train_df, model_name, grid_params):
    """
    Inner 5-fold CV for Grid Search
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = 0
    best_params = None
    
    for config in generate_grid(grid_params):
        scores = []
        
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(X, y)):
            # Train: AUGMENTED
            train_inner = AugmentedFCDataset(train_df.iloc[train_idx])
            # Val: ORIGINAL
            val_inner = FCDataset(train_df.iloc[val_idx])
            
            model = train_single_fold(train_inner, val_inner, config)
            acc = evaluate(model, val_inner)
            scores.append(acc)
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            
### **6.3 CSV Dependency Removal**
-  Removed: `resolve_split_csv_path()` from `cnn_utils.py`
-  Used: Dynamic splitting in memory with StratifiedKFold

---

## LITERATURE REFERENCES

- **AlexNet**: Krizhevsky et al., "ImageNet Classification with Deep CNNs", 2012
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition", 2015
- **VGG16**: Simonyan & Zisserman, "Very Deep CNNs for Large-Scale Image Recognition", 2014

---

**CONFIGURATION:**
*   `src/DL_analysis/config/cnn.json`: Unified Config (Jobs, Env Paths, Output Dir)
*   `src/DL_analysis/config/cnn_grid.json`: Hyperparameters (LR, Epochs, etc.)
