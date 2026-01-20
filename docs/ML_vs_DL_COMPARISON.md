# Machine Learning vs Deep Learning: Complete Comparison

## Executive Summary

Questo documento confronta in dettaglio gli approcci **Machine Learning (ML)** e **Deep Learning (DL)** implementati nel progetto, evidenziando differenze strutturali, metodologiche e operative.

---

## 1. Architettura dei Dati

### 1.1 Machine Learning
- **Input**: Features pre-calcolate (CSV)
  - Connettività funzionale (FDC)
  - Network measures (7 networks × 3 metriche = 21 features)
  - UMAP embedding (dimensionalità ridotta)
- **Dimensione**: ~100-150 features per soggetto
- **Formato**: Tabellare (pandas DataFrame)
- **Preprocessing**: UMAP fit solo su train, transform su test

### 1.2 Deep Learning
- **Input**: Mappe 3D di connettività funzionale raw
  - Volumi NIfTI (91 × 109 × 91 voxel)
  - Convertiti in `.npy` per efficienza
- **Dimensione**: ~900.000 voxel per soggetto
- **Formato**: Tensori 4D (batch, channel, depth, height, width)
- **Preprocessing**: Normalizzazione, augmentation via bootstrapping HCP

### Differenza Chiave
- **ML**: Feature engineering manuale → input compresso
- **DL**: Feature learning automatico → input raw ad alta dimensionalità

---

## 2. Gestione Train/Test Split

### 2.1 Machine Learning
```
Dataset completo (130 pazienti)
    ↓
CSV pre-diviso: train (104) / test (26)
    ↓
Split FISSO per tutti gli esperimenti
    ↓
Seed loop: 5 seeds × stesso split
```

**Caratteristiche**:
- Split unico da `ADNI_PSP_splitted.csv`
- Test set **sempre uguale** (stessi 26 pazienti)
- Seeds controllano solo inizializzazione modello
- Std(accuracy) = variabilità da seed, NON da split

### 2.2 Deep Learning
```
Dataset completo (130 pazienti)
    ↓
CSV pre-diviso: train (104) / test (26)
    ↓
Split FISSO per tutti gli esperimenti
    ↓
5-fold CV SUL TRAIN SET (104) → 5 modelli
    ↓
Best fold checkpoint → test set (26)
```

**Caratteristiche**:
- Split unico da `ADNI_PSP_splitted_cnn.csv`
- Test set **sempre uguale** (stessi 26 pazienti)
- CV **solo su train** per selezionare miglior fold
- Seeds controllano split dei fold + inizializzazione

### Differenza Chiave
- **ML**: Train intero → test
- **DL**: Train → 5-fold CV (selezione) → best checkpoint → test

---

## 3. Cross-Validation

### 3.1 Machine Learning (Inner CV)
```python
# GridSearchCV con StratifiedKFold (n_splits=5)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    scoring='accuracy'
)
grid_search.fit(x_train, y_train)  # Solo su train (104)
best_model = grid_search.best_estimator_  # Refitted su train intero
```

**Scopo**: Hyperparameter selection
**Output**: Modello refittato su train completo con best params
**Test set**: MAI usato durante CV

### 3.2 Deep Learning (Fold Selection CV)
```python
# StratifiedKFold su train set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_subjects, train_labels)):
    # Train su fold_train, valida su fold_val
    model = train_on_fold(fold_train, fold_val)
    save_checkpoint(f"best_model_fold{fold}.pt")

# Seleziona best fold basato su val_accuracy
best_checkpoint = select_best_fold(all_folds)
```

**Scopo**: Model selection (best fold)
**Output**: Checkpoint del fold con migliore val_accuracy
**Test set**: Valutato DOPO con best checkpoint

### Differenza Chiave
- **ML**: CV per hyperparameter, poi refit su train intero
- **DL**: CV per selezione fold, poi usa best checkpoint (NON refit)

---

## 4. Hyperparameter Tuning

### 4.1 Machine Learning
```python
# Automatico con GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search.fit(x_train, y_train)
# Output: best_params_ automatico
```

**Metodo**: GridSearchCV integrato
**Search Space**: Definito in `ml_config.json`
**Valutazione**: Media accuracy su 5-fold CV interno
**Refit**: Automatico su train intero con best params

### 4.2 Deep Learning
```python
# Manuale con loop su grid
grid = {
    'model_type': ['resnet', 'densenet', 'vgg16'],
    'batch_size': [4, 8, 16],
    'lr': [1e-3, 1e-4],
    'weight_decay': [0.001]
}

for combo in itertools.product(*grid.values()):
    if is_valid_combo(combo):
        result = train_with_cv(combo)  # 5-fold CV
        save_result(combo, result)

# Output: grid_results.csv con tutte le config
```

**Metodo**: Loop manuale + `itertools.product()`
**Search Space**: Definito in `cnn_grid.json`
**Valutazione**: Avg accuracy su 5 fold, best fold accuracy
**Refit**: NON necessario, usa best fold checkpoint

### Differenza Chiave
- **ML**: Tuning automatico con refit
- **DL**: Tuning manuale, selezione fold-based

---

## 5. Modelli

### 5.1 Machine Learning
```python
# Scikit-learn classifiers
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNeighbors': KNeighborsClassifier()
}
```

**Tipo**: Tree-based + KNN
**Input**: Vettore features (100-150 dimensioni)
**Parametri**: ~100-1000 (RF/GB trees)
**Training**: CPU-based, veloce (secondi)

### 5.2 Deep Learning
```python
# PyTorch 3D CNNs
models = {
    'ResNet3D': r3d_18 (torchvision),
    'DenseNet3D': DenseNet121 (MONAI),
    'VGG16_3D': Custom implementation
}
```

**Tipo**: Convolutional Neural Networks 3D
**Input**: Tensore 4D (1, 91, 109, 91)
**Parametri**: ~1-30 milioni (CNN layers)
**Training**: GPU-based, lento (ore)

### Differenza Chiave
- **ML**: Shallow models, feature-based
- **DL**: Deep models, imparano features da raw data

---

## 6. Data Augmentation

### 6.1 Machine Learning
```python
# NO augmentation
# Train set: 104 soggetti
# Ogni soggetto = 1 sample
```

**Augmentation**: Nessuna
**Train size**: 104 samples

### 6.2 Deep Learning
```python
# HCP bootstrapping augmentation
# Train set: 104 soggetti
# Ogni soggetto → 50 varianti (diversi subset HCP)
# Augmented train size: 104 × 50 = 5200 samples
```

**Augmentation**: Bootstrapping su SCA (HCP subjects)
**Train size**: ~5200 samples (50× augmentation)
**Implementation**: `AugmentedFCDataset` class

### Differenza Chiave
- **ML**: Train originale (104 samples)
- **DL**: Train augmented (5200 samples)

---

## 7. Output e Metriche

### 7.1 Machine Learning
```
results/ml_analysis/
├── classification_results.csv
│   ├── Seed: 42, 123, 2023, 31415, 98765
│   ├── Model: RF, GB, KNN
│   ├── Metrics: Acc, Prec, Rec, F1, AUC-ROC
│   └── Format: 1 riga per (seed, model)
└── plots/
    ├── confusion_matrices/
    └── feature_importance/ (RF only)
```

**Output Principale**:
- CSV singolo con tutti i risultati
- Media ± std su 5 seeds
- Esempio: `Accuracy: 65.3% ± 0.2%`

### 7.2 Deep Learning
```
results/runs/
├── all_training_results.csv
│   ├── Run ID: 1-5 (one per seed)
│   ├── Best fold: 1-5
│   ├── Metrics: Best acc, Avg acc, Train loss, Val loss
│   └── Hyperparams: model, lr, batch_size, etc.
├── all_testing_results.csv
│   ├── Run ID: 1-5
│   ├── Metrics: Acc, Prec, Rec, F1 on test set
│   └── Seed info merged
└── run{1-5}/
    ├── best_model_fold{1-5}.pt (checkpoints)
    ├── training_folds.xlsx (per-epoch)
    └── plots/ (learning curves, confusion matrix)
```

**Output Principale**:
- 2 CSV: training (CV) + testing (test set)
- Checkpoints salvati per ogni fold
- Excel con metriche per-epoch

### Differenza Chiave
- **ML**: Output compatto, focus su metriche finali
- **DL**: Output dettagliato, checkpoints + logs

---

## 8. Workflow Completo

### 8.1 Machine Learning
```
┌─────────────────────────────────────────────────────┐
│ 1. Load CSV split (train/test)                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 2. FOR seed in [42, 123, 2023, 31415, 98765]:     │
│    ├─ Set seed                                      │
│    ├─ Run UMAP (fit on train, transform test)      │
│    ├─ FOR model in [RF, GB, KNN]:                 │
│    │   ├─ IF tuning:                               │
│    │   │   └─ GridSearchCV (inner 5-fold on train) │
│    │   ├─ Train on full train (104)                │
│    │   ├─ Test on test (26)                        │
│    │   └─ Save metrics                             │
│    └─ (Optional) Permutation test                  │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 3. Aggregate results: Mean ± Std across seeds      │
└─────────────────────────────────────────────────────┘
```

### 8.2 Deep Learning
```
┌─────────────────────────────────────────────────────┐
│ 1. Load CSV split (train/test)                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 2. FOR seed in [42, 123, 2023, 31415, 98765]:     │
│    ├─ Set seed                                      │
│    ├─ 5-Fold CV on train (104):                    │
│    │   └─ FOR fold in 1..5:                        │
│    │       ├─ Train on fold_train (~83)            │
│    │       ├─ Validate on fold_val (~21)           │
│    │       └─ Save checkpoint                      │
│    ├─ Select best fold (max val_accuracy)          │
│    └─ Save training summary                        │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 3. FOR each run (seed):                            │
│    ├─ Load best fold checkpoint                    │
│    ├─ Evaluate on test (26)                        │
│    └─ Save testing summary                         │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 4. Aggregate: Mean ± Std across runs               │
└─────────────────────────────────────────────────────┘
```

---

## 9. Reproducibilità

### 9.1 Machine Learning
```python
# Seed control points:
1. np.random.seed(seed)         # UMAP, train/test split
2. random_state in GridSearchCV  # CV folds
3. random_state in model         # Model initialization
```

**Risultato**: Stesso seed → risultati identici

### 9.2 Deep Learning
```python
# Seed control points:
1. random.seed(seed)                      # Python random
2. np.random.seed(seed)                   # NumPy
3. torch.manual_seed(seed)                # PyTorch CPU
4. torch.cuda.manual_seed_all(seed)       # PyTorch GPU
5. torch.backends.cudnn.deterministic=True # CUDNN
6. StratifiedKFold(random_state=seed)     # CV folds
```

**Risultato**: Stesso seed → fold splits identici, ma risultati possono variare leggermente (GPU non-determinism)

---

## 10. Complessità Computazionale

### 10.1 Machine Learning
- **Preprocessing**: ~1 min (UMAP fit su 104 samples)
- **Training per seed**: ~10 sec (RF/GB) o <1 sec (KNN)
- **Tuning (GridSearchCV)**: ~5 min (dipende da grid size)
- **Testing**: <1 sec
- **Total per experiment**: ~10-15 min (5 seeds × 3 models)
- **Hardware**: CPU sufficiente

### 10.2 Deep Learning
- **Preprocessing**: ~30 min (convert NIfTI → .npy, augmentation)
- **Training per fold**: ~30-60 min (50 epochs, GPU)
- **Training per seed (5 folds)**: ~2.5-5 ore
- **Tuning (grid search)**: ~50 ore (18 configs × 5 folds × 1 ora)
- **Testing**: ~5 min
- **Total per experiment**: ~12-25 ore (5 seeds)
- **Hardware**: GPU necessaria (8-16 GB VRAM)

### Differenza Chiave
- **ML**: Veloce, scalabile, CPU-based
- **DL**: Lento, resource-intensive, GPU-based

---

## 11. Vantaggi e Svantaggi

### 11.1 Machine Learning

#### Vantaggi
✅ Veloce e efficiente  
✅ Interpretabile (feature importance)  
✅ Pochi hyperparameters da ottimizzare  
✅ Non richiede GPU  
✅ Robusto su dataset piccoli (<1000 samples)  
✅ Facile debugging  

#### Svantaggi
❌ Richiede feature engineering manuale  
❌ Perde informazione spaziale (UMAP dimensionality reduction)  
❌ Limitato su dati raw ad alta dimensionalità  
❌ Non impara features gerarchiche  

### 11.2 Deep Learning

#### Vantaggi
✅ Feature learning automatico  
✅ Preserva informazione spaziale 3D  
✅ Scalabile su dataset grandi  
✅ State-of-the-art su imaging medicale  
✅ Transfer learning possibile (ImageNet pretrained)  

#### Svantaggi
❌ Lento e computazionalmente costoso  
❌ Richiede GPU potente  
❌ Black box (difficile interpretazione)  
❌ Prone to overfitting su dataset piccoli (<5000 samples)  
❌ Molti hyperparameters da ottimizzare  
❌ Debugging complesso  

---

## 12. Raccomandazioni per Allineamento

### 12.1 Differenze Critiche da Allineare

#### Issue 1: CV Usage
- **ML**: CV per hyperparameter selection + refit su train intero
- **DL**: CV per fold selection + usa best checkpoint (no refit)

**Raccomandazione**: 
- Mantenere approcci separati (sono filosoficamente diversi)
- Documentare chiaramente che DL usa "best fold model", non "retrained model"

#### Issue 2: Seed Interpretation
- **ML**: Seed = solo inizializzazione modello (split fisso)
- **DL**: Seed = fold splits + inizializzazione modello

**Raccomandazione**:
- In DL, usare seed **fisso** per fold split, seed **variabile** solo per weight init
- Modificare codice:
  ```python
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Fixed
  torch.manual_seed(seed)  # Variable
  ```

#### Issue 3: Output Format
- **ML**: CSV singolo, media ± std
- **DL**: 2 CSV (training + testing), no aggregazione automatica

**Raccomandazione**:
- Aggiungere script post-processing per DL che calcoli media ± std su seeds
- Formato output: `test_accuracy: 70.2% ± 1.5%` (come ML)

#### Issue 4: Test Set Contamination
- **ML**: Test mai usato in training/tuning ✅
- **DL**: Test mai usato in training/tuning ✅
- **Entrambi corretti!**

### 12.2 Best Practices Comuni

#### ✅ Train/Test Split
- Mantenere **split fisso** da CSV per entrambi
- Stessi pazienti in train/test per confronto equo

#### ✅ Stratification
- Entrambi usano `StratifiedKFold` / `StratifiedShuffleSplit`
- Garantisce bilanciamento classi

#### ✅ Metrics
- Stesse metriche: Accuracy, Precision, Recall, F1
- Aggiungere AUC-ROC anche in DL

#### ✅ Reproducibility
- Seed control su tutti i RNG
- Documentare hardware (CPU/GPU) per replicabilità

---

## 13. Conclusioni

### Quando Usare ML
- Dataset piccolo (<1000 samples) ✅ (130 in questo caso)
- Features già disponibili/facili da calcolare
- Interpretabilità richiesta
- Risorse limitate (no GPU)
- Prototipazione rapida

### Quando Usare DL
- Dataset grande (>5000 samples) ❌ (130 in questo caso)
- Dati raw (immagini, volumi 3D)
- Feature learning automatico desiderato
- GPU disponibile
- State-of-the-art performance richiesta

### Per Questo Progetto
- **Dataset size**: 130 soggetti → ML più appropriato
- **Data augmentation**: DL compensa con 50× augmentation
- **Interpretazione**: ML vince (feature importance)
- **Performance**: Da confrontare empiricamente (ML vs DL accuracy)

### Strategia Finale
1. **Mantenere entrambi gli approcci** (complementari)
2. **Allineare output format** per confronto diretto
3. **Documentare differenze** (CV philosophy)
4. **Confronto finale**: ML baseline vs DL con augmentation
5. **Ensemble?**: Combinare predizioni ML + DL per robustezza

---

## Appendice: Checklist Allineamento

### ✅ Completati
- [x] Split fisso train/test (CSV pre-diviso)
- [x] Stesse metriche (Acc, Prec, Rec, F1)
- [x] Seed control per reproducibilità
- [x] Test set mai usato in training

### ⏳ Da Implementare
- [ ] DL: Seed fisso per fold split, variabile per init
- [ ] DL: Script post-processing per media ± std
- [ ] DL: Aggiungere AUC-ROC metric
- [ ] ML/DL: Unified output CSV format
- [ ] ML/DL: Confronto side-by-side nello stesso notebook

### 📊 Da Analizzare
- [ ] ML accuracy vs DL accuracy (quale performa meglio?)
- [ ] Effetto augmentation (DL con/senza)
- [ ] Tempo computazionale vs performance gain
- [ ] Feature importance (ML) vs Grad-CAM (DL)
