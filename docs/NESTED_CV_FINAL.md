# NESTED CROSS-VALIDATION - IMPLEMENTAZIONE DL/ML

**Data:** 20 Gennaio 2026  
**Versione:** 1.0 - Documento Finale

---

## 1. ARCHITETTURA NESTED CV

```
┌──────────────────────────────────────────────────────────────┐
│ OUTER CROSS-VALIDATION (5-fold StratifiedKFold)             │
│ • Seed fisso: 42 (riproducibilità)                          │
│ • Split: 80% Train (~105 pz) / 20% Test (~27 pz)           │
│ • Per ogni fold: train diverso, test diverso                │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ FOLD 1 OUTER                                                 │
│ Train: ~84 soggetti | Test: ~21 soggetti                    │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ INNER CROSS-VALIDATION (5-fold StratifiedKFold)             │
│ • Su SOLO Train Outer (~84 pz)                              │
│ • Grid Search: N configurazioni di iperparametri            │
│ • Validation: fold interno (dati ORIGINALI)                 │
│ • Seed: 42 (stesso outer per riproducibilità)               │
└──────────────────────────────────────────────────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │ Inner Fold 1: Train ~67 | Val ~17   │
         │ Inner Fold 2: Train ~67 | Val ~17   │
         │ Inner Fold 3: Train ~67 | Val ~17   │
         │ Inner Fold 4: Train ~67 | Val ~17   │
         │ Inner Fold 5: Train ~67 | Val ~17   │
         └──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ SELEZIONE BEST HYPERPARAMETERS                               │
│ • Media accuracy su 5 inner folds                            │
│ • Configurazione con accuracy maggiore → BEST PARAMS         │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ FULL RETRAIN (con best params)                              │
│ • Train: TUTTI gli 84 pz del fold esterno                   │
│ • Dati AUGMENTATI: 10× HCP bootstrap                        │
│ • Epochs: fino a convergenza (con early stopping)            │
└──────────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ TESTING FOLD 1                                               │
│ • Test: 21 pz del fold esterno                              │
│ • Dati ORIGINALI (NO augmentation)                          │
│ • Metriche: Accuracy, F1, Precision, Recall, AUC            │
└──────────────────────────────────────────────────────────────┘
                            ▼
       [RIPETI per FOLD 2, 3, 4, 5 OUTER]
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ AGGREGAZIONE RISULTATI FINALI                                │
│ • Mean ± Std su 5 outer folds                                │
│ • Per-fold metrics (tabella completa)                        │
│ • Best model selection: fold con accuracy maggiore           │
│ • Salvataggio: results/nested_cv/GROUP/aggregated.json      │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. DATA AUGMENTATION (10× HCP BOOTSTRAP)

**POLICY FONDAMENTALE:**
- ✅ **AUGMENTATION**: SOLO nei set di TRAINING (outer train, inner train durante grid search, full retrain)
- ❌ **NO AUGMENTATION**: SEMPRE in validation e test (dati originali)

**Dataset Classes:**
- `FCDataset`: dati originali (validation e test)
- `AugmentedFCDataset`: 10× augmented (training)

**Integrazione con Dynamic Splitting:**
- Durante Outer CV: `AugmentedFCDataset` carica da `data/FCmaps_augmented_processed/`
- Ogni soggetto ha 10 file `.npy` (10 bootstrap HCP diversi)
- Validation/Test: `FCDataset` carica da `data/FCmaps_processed/` (originali)

**Modifica necessaria:**
- ❌ NON serve soluzione ibrida
- ✅ Dynamic splitting diretto: `StratifiedKFold` su lista soggetti in memoria
- ✅ `datasets.py` già compatibile: accetta DataFrame con ID, label

---

## 3. GRID SEARCH E INNER CV

**GRID SEARCH PLACEMENT:**
- Grid search: DENTRO outer loop
- Eseguito su: Train Outer (~84 pz)
- Inner CV: 5-fold per validation
- NO data leakage: test outer MAI visto durante grid search

---

## 4. MODELLI E IPERPARAMETRI DA LETTERATURA

### **AlexNet3D** (NEW - da Paper Originale Krizhevsky 2012)

**Architettura:**
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

**Iperparametri Originali (Paper):**
- Learning Rate: **0.01**
- Batch Size: **128**
- Momentum: **0.9**
- Weight Decay: **0.0005**
- Dropout: **0.5**
- Epochs: **90**
- Optimizer: **SGD + Momentum**

**Adattamenti (Grid Search - Single Value):**
```python
{
    'lr': [1e-2],                 # Fixed 0.01 (da valutare)
    'batch_size': [8],             # Fixed (constraint)
    'weight_decay': [5e-4],        # Original
    'epochs': [40],                # Reduced
    'patience': [15]               # Early Stopping
}
```

---

### **ResNet18** (r3d_18 - da Paper He 2015)

**Architettura:**
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

**Iperparametri Originali (Paper):**
- Learning Rate: **0.1** (0.01 warmup)
- Batch Size: **256**
- Momentum: **0.9**
- Weight Decay: **0.0001**
- NO Dropout
- Epochs: **60**
- Optimizer: **SGD + Momentum**

**Adattamenti (Grid Search - Single Value):**
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

### **VGG16_3D** (da Paper Simonyan 2014)

**Architettura:**
```python
# Configurazione D (VGG16)
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

**Iperparametri Originali (Paper):**
- Learning Rate: **0.01**
- Batch Size: **256**
- Momentum: **0.9**
- Weight Decay: **5e-4**
- Dropout: **0.5**
- Epochs: **74**
- Optimizer: **SGD + Momentum**

**Adattamenti (Grid Search - Single Value):**
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

## 5. RISULTATI AGGREGATI

**Per ogni Outer Fold (5 totali):**
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

**Aggregazione Finale:**
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

**Metriche Riportate:**
- **Mean ± Std**: Accuracy, F1, Precision, Recall, AUC su 5 outer folds
- **Per-fold**: Tabella completa con tutte le metriche
- **Best Hyperparams**: Frequenza di selezione per ogni parametro
- **Best Model**: Fold con performance migliore (per deployment)

**Struttura Output:**
```
results/nested_cv/
├── ADNI_PSP/
│   ├── fold_1/
│   │   ├── best_model.pt
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   ├── fold_2/ ... fold_5/
│   └── aggregated_results.json
├── ADNI_CBS/ (stessa struttura)
└── PSP_CBS/ (stessa struttura)

---


## 6. MODIFICHE CODICE NECESSARIE

### **6.1 Nuovo File: `nested_cv_runner.py`**

def nested_cv_classification(group1, group2, model_name, config_path):
    """
    Nested CV: Outer 5-fold + Inner 5-fold Grid Search
    """
    # 1. Carica metadata
    df_meta = load_metadata(group1, group2)
    
    # 2. Outer CV: StratifiedKFold(5, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n=== OUTER FOLD {outer_fold+1}/5 ===")
        
        # 3. Split outer
        train_subjects = df_meta.iloc[train_idx]
        test_subjects = df_meta.iloc[test_idx]
        
        # 4. Inner CV Grid Search (su train_subjects)
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
    Inner 5-fold CV per Grid Search
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
            
### **6.2 Modifica `datasets.py`** ✅ GIÀ COMPATIBILE
- `FCDataset` e `AugmentedFCDataset` accettano DataFrame
- Nessuna modifica necessaria

### **6.3 Eliminare Dipendenza CSV**
- ❌ Rimuovi: `resolve_split_csv_path()` da `cnn_utils.py`
- ✅ Usa: Dynamic splitting in memoria con StratifiedKFold

### **7.4 Aggiorna `models.py`**
- ✅ Aggiungi: Classe `AlexNet3D` (vedi architettura sopra)
- ✅ Modifica: `ResNet3D` usa `r3d_18` invece di `r3d_34`
- ✅ Mantieni: `VGG16_3D` invariato

---

## RIFERIMENTI LETTERATURA

- **AlexNet**: Krizhevsky et al., "ImageNet Classification with Deep CNNs", 2012
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition", 2015
- **VGG16**: Simonyan & Zisserman, "Very Deep CNNs for Large-Scale Image Recognition", 2014

---

**PROSSIMO STEP:** Conferma punti aperti (domande 1-5) e inizio implementazione `nested_cv_runner.py`.
