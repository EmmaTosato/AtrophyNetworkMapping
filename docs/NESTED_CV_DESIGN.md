# 🔄 Pipeline Restructuring: Nested Cross-Validation for DL & ML

## 📋 Executive Summary

**Obiettivo**: Implementare **Nested 5-Fold Cross-Validation** per entrambe le pipeline (DL e ML) con identica struttura.

**Struttura Proposta**:
```
OUTER CV (5-fold) → Train/Test Split Diversi
    └─> INNER CV (5-fold) → Hyperparameter Tuning
        └─> Full Retrain → Test su fold esterno
```

---

## 1️⃣ ANALISI DELLA TUA PROPOSTA

### ✅ Proposta Corretta

```
Outer 5-Fold CV:
  ├─ Fold 1: Train (80%) / Test (20%)
  │    └─> Inner 5-Fold CV su Train:
  │         ├─ Grid Search per hyperparameter tuning
  │         ├─ Seleziona best hyperparams
  │         └─> Full Retrain su Train completo
  │              └─> Test su Test del fold esterno
  ├─ Fold 2: ...
  ├─ Fold 3: ...
  ├─ Fold 4: ...
  └─ Fold 5: ...
```

**Risultato Finale**: 5 valutazioni indipendenti (una per fold esterno)

### 🎯 Risposte alle Tue Domande

#### Q1: "Ho un'impostazione corretta?"
✅ **SÌ, PERFETTA!** 

Questa è la **Nested Cross-Validation** standard:
- **Outer CV**: Valuta performance generalizzata (unbiased estimate)
- **Inner CV**: Trova best hyperparameters per quel fold
- **Full retrain**: Massimizza uso dei dati

#### Q2: "I dati augmentati dove?"
📍 **Risposta**:
- **Outer Train**: Dati augmentati (50×)
- **Inner Train (dentro CV)**: Dati augmentati (50×)
- **Inner Val (dentro CV)**: Dati originali (NO augmentation)
- **Outer Test**: Dati originali (NO augmentation)

**Regola**: Augmentation **SOLO** nei set di training, MAI in validation/test

#### Q3: "La grid search dovrebbe avviarsi dal loop esterno?"
✅ **SÌ!** La grid search è parte dell'**Inner CV** che si ripete **per ogni fold esterno**.

```python
for outer_fold in outer_cv.split():  # OUTER LOOP
    X_train_outer, X_test_outer = ...
    
    # INNER CV (Grid Search)
    grid = GridSearchCV(model, params, cv=5)  # <-- Qui!
    grid.fit(X_train_outer)
    best_params = grid.best_params_
    
    # Full retrain con best params
    model.set_params(**best_params)
    model.fit(X_train_outer)
    
    # Test su fold esterno
    y_pred = model.predict(X_test_outer)
```

#### Q4: "È possibile mantenere l'organizzazione attuale?"
✅ **SÌ, con modifiche minime!**

**Cosa mantenere**:
- Struttura cartelle `src/DL_analysis/` e `src/ML_analysis/`
- Dataset classes (`FCDataset`, `AugmentedFCDataset`)
- Models (`ResNet3D`, `AlexNet3D`, `VGG16_3D`)
- Utility functions (`cnn_utils.py`, `ml_utils.py`)

**Cosa modificare**:
- `run.py`: Aggiungere outer loop
- `hyper_tuning.py`: Integrare in outer loop (o separare inner CV)
- Split system: Generare split al volo invece di CSV fissi

---

## 2️⃣ DESIGN DETTAGLIATO: NESTED CV PIPELINE

### Architettura Proposta

```
┌─────────────────────────────────────────────────────────────┐
│  OUTER CROSS-VALIDATION (5-fold StratifiedKFold)            │
│  • Seed fisso (42) per riproducibilità                      │
│  • Split: 80% Train / 20% Test                             │
└─────────────────────────────────────────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  FOLD 1                      │
         │  Train: ~84 pz, Test: ~21 pz │
         └──────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  INNER CROSS-VALIDATION (5-fold)                │
    │  Su Train (~84 pz) del fold esterno             │
    │  • Grid Search hyperparameters                  │
    │  • Validation su fold interno                   │
    └─────────────────────────────────────────────────┘
                        ▼
         ┌────────────────────────────┐
         │  Best Hyperparameters       │
         │  (dal grid search)          │
         └────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  FULL RETRAIN                                   │
    │  • Train su TUTTI gli 84 pz (train esterno)    │
    │  • Usa best hyperparams da inner CV             │
    │  • Dati augmentati (50×) nel training           │
    └─────────────────────────────────────────────────┘
                        ▼
         ┌────────────────────────────┐
         │  TESTING                   │
         │  Test sui 21 pz (fold est.)│
         │  Dati originali (NO aug)   │
         └────────────────────────────┘
                        ▼
         [Ripeti per FOLD 2, 3, 4, 5]
                        ▼
    ┌─────────────────────────────────────────────────┐
    │  AGGREGAZIONE RISULTATI                         │
    │  • Mean ± Std su 5 fold                        │
    │  • Per-fold results                             │
    └─────────────────────────────────────────────────┘
```

---

## 3️⃣ DATA AUGMENTATION STRATEGY

### Dove Usare Augmentation

```
Outer Fold 1:
  ├─ Train (~84 pz)
  │   ├─ Inner Fold 1.1:
  │   │   ├─ Train (~67 pz) → AUGMENTED (50×)  ✅
  │   │   └─ Val (~17 pz) → ORIGINAL           ✅
  │   ├─ Inner Fold 1.2:
  │   │   ├─ Train (~67 pz) → AUGMENTED (50×)  ✅
  │   │   └─ Val (~17 pz) → ORIGINAL           ✅
  │   └─ [... altri fold ...]
  │
  ├─ Full Retrain (84 pz) → AUGMENTED (50×)    ✅
  └─ Test (~21 pz) → ORIGINAL                  ✅
```

**Regola d'oro**: 
- 🟢 **TRAINING**: Sempre augmented
- 🔴 **VALIDATION/TEST**: Sempre original

---

## 4️⃣ SPLIT MANAGEMENT: CSV FISSI vs DYNAMIC

### ❌ Vecchio Approccio (Problematico per Nested CV)

```python
# ml_split.json genera CSV fisso
split_csv = "ADNI_PSP_splitted.csv"  # 105 train, 27 test (fisso)

# Problema: Nested CV richiede split DIVERSI ogni fold!
```

### ✅ Nuovo Approccio: Dynamic Splitting

```python
# Load metadata
df_labels = pd.read_csv("assets/metadata/labels.csv")
df_pair = df_labels[df_labels["Group"].isin([group1, group2])]

# Outer CV con seed fisso
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(subjects, labels)):
    # Split dinamico per questo fold
    train_df = df_pair.iloc[train_idx]
    test_df = df_pair.iloc[test_idx]
    
    # Inner CV su train_df
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # ... grid search ...
```

**Vantaggi**:
- ✅ Nessun CSV fisso da gestire
- ✅ Riproducibile (seed=42)
- ✅ Flessibile (cambi n_folds facilmente)

**Svantaggi**:
- ⚠️ Non puoi comparare con vecchi risultati (split diversi)

### 🔀 Soluzione Ibrida (Raccomandato)

```python
if params.get("use_nested_cv", False):
    # Nested CV: dynamic splits
    outer_cv = StratifiedKFold(...)
else:
    # Fixed split: backward compatibility
    split_csv = resolve_split_csv_path(...)
```

---

## 5️⃣ GRID SEARCH: OUTER vs INNER

### ❌ SBAGLIATO: Grid Search nell'Outer Loop

```python
# SBAGLIATO!
for outer_fold in outer_cv:
    # Grid search QUI è FUORI dal fold!
    grid = GridSearchCV(...).fit(ALL_DATA)  # ❌ Data leakage!
    
    X_train, X_test = split_fold(outer_fold)
    grid.best_estimator_.predict(X_test)  # ❌ Bias!
```

### ✅ CORRETTO: Grid Search nell'Inner Loop

```python
# CORRETTO!
for outer_fold in outer_cv:
    X_train_outer, X_test_outer = split_fold(outer_fold)
    
    # Grid search SOLO su train outer (no leakage!)
    inner_cv = StratifiedKFold(5)
    grid = GridSearchCV(model, params, cv=inner_cv)
    grid.fit(X_train_outer, y_train_outer)  # ✅
    
    best_params = grid.best_params_
    
    # Full retrain con best params
    final_model = Model(**best_params)
    final_model.fit(X_train_outer, y_train_outer)
    
    # Test su outer fold
    y_pred = final_model.predict(X_test_outer)  # ✅ Unbiased!
```

---

## 6️⃣ STRUTTURA CARTELLE MODIFICATA

### Proposta Output Directory

```
results/
├── nested_cv/
│   ├── ADNI_vs_PSP/
│   │   ├── fold_1/
│   │   │   ├── inner_cv_results.csv      # Grid search
│   │   │   ├── best_hyperparams.json
│   │   │   ├── best_model.pt             # Full retrain
│   │   │   ├── test_results.csv
│   │   │   └── confusion_matrix.png
│   │   ├── fold_2/
│   │   │   └── ...
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   ├── fold_5/
│   │   ├── summary_all_folds.csv         # Aggregato
│   │   └── mean_std_metrics.csv
│   │
│   ├── ADNI_vs_CBS/
│   │   └── ...
│   └── PSP_vs_CBS/
│       └── ...
│
└── old_runs/  # Backup vecchi risultati
    └── ...
```

---

## 7️⃣ MODELS: CAMBIO ARCHITETTURE

### ❌ Vecchi Modelli
- DenseNet3D (MONAI)

### ✅ Nuovi Modelli

#### 1. AlexNet3D
```python
class AlexNet3D(nn.Module):
    """
    3D adaptation of AlexNet
    Input: (B, 1, 91, 109, 91)
    
    Literature:
    - Original: Krizhevsky et al., 2012 (ImageNet)
    - 3D Medical: Hosseini-Asl et al., 2016 (Alzheimer's MRI)
    
    Recommended Hyperparams:
    - lr: 1e-3 (Adam) / 1e-2 (SGD + momentum 0.9)
    - batch_size: 16-32
    - weight_decay: 5e-4
    - dropout: 0.5
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: 11×11×11, stride 4
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Conv2: 5×5×5
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Conv3-5: 3×3×3
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 3 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
```

#### 2. VGG16_3D (già presente, mantieni)
```python
# Recommended Hyperparams (letteratura):
# - lr: 1e-4 (Adam) / 1e-3 (SGD + momentum 0.9)
# - batch_size: 8-16 (GPU memory intensive)
# - weight_decay: 5e-4
# - dropout: 0.5
```

#### 3. ResNet18_3D
```python
class ResNet18_3D(nn.Module):
    """
    3D ResNet-18 (instead of ResNet-50)
    
    Literature:
    - Original: He et al., 2015 (ImageNet)
    - 3D Medical: Chen et al., 2019 (Medical Segmentation)
    
    Recommended Hyperparams:
    - lr: 1e-3 (Adam) / 1e-2 (SGD + momentum 0.9, weight_decay 1e-4)
    - batch_size: 16-32
    - weight_decay: 1e-4
    - No dropout (batch norm invece)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Usa torchvision.models.video.r3d_18
        from torchvision.models.video import r3d_18
        
        base_model = r3d_18(pretrained=False)
        
        # Modifica input channel (3 → 1)
        base_model.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        
        # Modifica output classes
        base_model.fc = nn.Linear(512, num_classes)
        
        self.model = base_model
    
    def forward(self, x):
        return self.model(x)
```

---

## 8️⃣ HYPERPARAMETERS DA LETTERATURA

### AlexNet3D
**References**:
- Hosseini-Asl et al. (2016) - "Alzheimer's Disease Diagnostics by Adaptation of 3D Convolutional Network"
- Suk et al. (2017) - "Deep Learning in Medical Image Analysis"

**Optimal Params**:
```json
{
  "lr": [1e-3, 1e-4],
  "batch_size": [16, 32],
  "optimizer": "adam",
  "weight_decay": [5e-4, 1e-3],
  "dropout": [0.5],
  "epochs": [50, 100]
}
```

### ResNet18_3D
**References**:
- He et al. (2015) - "Deep Residual Learning"
- Korolev et al. (2017) - "Residual and Plain CNN for Alzheimer's Disease"

**Optimal Params**:
```json
{
  "lr": [1e-3, 1e-2],
  "batch_size": [16, 32],
  "optimizer": ["adam", "sgd"],
  "weight_decay": [1e-4, 1e-3],
  "momentum": [0.9],  // Solo per SGD
  "epochs": [50, 100]
}
```

### VGG16_3D
**References**:
- Simonyan & Zisserman (2014) - "Very Deep Convolutional Networks"
- Payan & Montana (2015) - "Predicting Alzheimer's disease: a neuroimaging study"

**Optimal Params**:
```json
{
  "lr": [1e-4, 1e-5],
  "batch_size": [4, 8, 16],  // GPU memory bound
  "optimizer": "adam",
  "weight_decay": [5e-4, 1e-3],
  "dropout": [0.5],
  "epochs": [50, 100]
}
```

---

## 9️⃣ IMPLEMENTATION ROADMAP

### Phase 1: Core Restructuring (Week 1)

#### Step 1.1: Create Nested CV Runner
**File**: `src/DL_analysis/training/nested_cv_runner.py`

```python
def nested_cross_validation(params):
    """
    Main nested CV pipeline.
    
    Outer loop: 5-fold stratified
    Inner loop: Grid search with 5-fold CV
    Full retrain: Best params on full outer train
    """
    # Load data
    df_labels = pd.read_csv(params['labels_path'])
    df_pair = filter_groups(df_labels, params['group1'], params['group2'])
    
    # Outer CV
    outer_cv = StratifiedKFold(
        n_splits=params['n_outer_folds'],
        shuffle=True,
        random_state=42  # Fixed for reproducibility
    )
    
    all_fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(...)):
        print(f"\n{'='*60}")
        print(f"OUTER FOLD {fold_idx + 1}/{params['n_outer_folds']}")
        print(f"{'='*60}")
        
        # Split data
        train_df = df_pair.iloc[train_idx]
        test_df = df_pair.iloc[test_idx]
        
        # Inner CV (Grid Search)
        best_params = inner_cv_tuning(train_df, params)
        
        # Full retrain
        model = train_full_model(train_df, best_params, params)
        
        # Test
        metrics = test_model(model, test_df, params)
        
        # Save fold results
        save_fold_results(fold_idx, best_params, metrics, params)
        all_fold_results.append(metrics)
    
    # Aggregate results
    aggregate_and_save(all_fold_results, params)
```

#### Step 1.2: Inner CV Tuning
**File**: `src/DL_analysis/training/inner_cv_tuning.py`

```python
def inner_cv_tuning(train_df, params):
    """
    Grid search con inner CV.
    Returns: best_hyperparams dict
    """
    from itertools import product
    
    # Grid
    grid = {
        'model_type': params['model_types'],
        'lr': params['lr_grid'],
        'batch_size': params['batch_size_grid'],
        'weight_decay': params['weight_decay_grid']
    }
    
    # Inner CV
    inner_cv = StratifiedKFold(n_splits=5, random_state=42)
    
    best_score = -np.inf
    best_params = None
    
    for config in product(*grid.values()):
        config_params = dict(zip(grid.keys(), config))
        
        # Validate configuration
        if not is_valid_combo(config_params):
            continue
        
        # 5-fold CV
        scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(...):
            # Train
            model = create_model(config_params)
            train_inner_fold(model, train_idx=inner_train_idx)
            
            # Validate
            val_score = validate_inner_fold(model, val_idx=inner_val_idx)
            scores.append(val_score)
        
        # Average score
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = config_params
    
    return best_params
```

#### Step 1.3: Full Retrain
**File**: Modifica `src/DL_analysis/training/train.py`

```python
def train_full_model(train_df, best_params, params):
    """
    Train su TUTTO il train set del fold esterno
    con best hyperparams da inner CV.
    """
    # Dataset (augmented)
    train_dataset = AugmentedFCDataset(
        params['data_dir_augmented'],
        train_df,
        params['label_column']
    )
    
    # Model
    model = create_model_from_params(best_params)
    
    # Train (no validation, full train)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, best_params)
    
    for epoch in range(params['epochs']):
        train_loss = train_epoch(model, train_dataset, criterion, optimizer)
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f}")
    
    return model
```

### Phase 2: Model Updates (Week 1-2)

#### Step 2.1: Add AlexNet3D
**File**: `src/DL_analysis/cnn/models.py`

- Implementa classe `AlexNet3D`
- Aggiungi in `create_model()` switch case
- Update `is_valid_combo()` con constraints AlexNet

#### Step 2.2: Replace DenseNet → AlexNet
**Files**: 
- `cnn_config.json`: Rimuovi "densenet", aggiungi "alexnet"
- `cnn_grid.json`: Update grid con nuovi modelli

#### Step 2.3: Update ResNet34 → ResNet18
**File**: `src/DL_analysis/cnn/models.py`

- Modifica `ResNet3D` per usare `r3d_18` invece di `r3d_34`

### Phase 3: Testing & Validation (Week 2)

#### Step 3.1: Unit Tests
**File**: `test/test_nested_cv.py`

```python
def test_outer_cv_splits():
    """Verifica che outer CV generi split stratificati corretti"""
    pass

def test_inner_cv_no_leakage():
    """Verifica nessun data leakage tra inner train/val"""
    pass

def test_augmentation_policy():
    """Verifica augmentation solo in training"""
    pass
```

#### Step 3.2: Integration Test
- Run su dataset ridotto (20 soggetti)
- Verifica tempi
- Controlla output files

### Phase 4: Documentation (Week 2)

#### Step 4.1: Update User Guide
**File**: `docs/NESTED_CV_USER_GUIDE.md`

- Come configurare nested CV
- Differenze con old pipeline
- Interpretazione risultati

#### Step 4.2: Technical Documentation
**File**: `docs/NESTED_CV_TECHNICAL.md`

- Architettura dettagliata
- Pseudocode
- Flow diagrams

---

## 🔟 COMPATIBILITÀ CON ML PIPELINE

### Allineamento ML ↔ DL

```
ML Pipeline:
  ├─ Outer CV (5-fold) ✅ UGUALE
  ├─ Inner CV (Grid Search) ✅ UGUALE  
  ├─ Full Retrain ✅ UGUALE
  └─ Aggregazione ✅ UGUALE

Differenze:
  ├─ Dati: Features (UMAP) vs Raw 3D volumes
  ├─ Models: Sklearn vs PyTorch
  └─ Augmentation: No vs 50× bootstrap
```

**Vantaggio**: Risultati direttamente comparabili!

---

## 🎯 DELIVERABLES FINALI

### Output Files

```
results/nested_cv/ADNI_vs_PSP/
├── summary_all_folds.csv
│   Columns: fold, model, lr, batch_size, accuracy, precision, recall, f1
│
├── mean_std_metrics.csv
│   Columns: model, metric, mean, std
│   Example: AlexNet3D, accuracy, 0.72, 0.05
│
└── fold_X/
    ├── inner_cv_grid_search.csv
    ├── best_hyperparams.json
    ├── best_model_fold{X}.pt
    ├── test_results.csv
    └── confusion_matrix.png
```

---

## ✅ NEXT STEPS

1. **Review questo documento** e approva design
2. **Decidi**: Mantieni backward compatibility con vecchia pipeline?
3. **Priorità modelli**: Quale implementare prima? (Raccomando: AlexNet3D)
4. **Timeline**: Conferma 2 settimane realistico?

---

**Vuoi che proceda con l'implementazione?** Posso:
1. Creare `nested_cv_runner.py`
2. Implementare `AlexNet3D`
3. Modificare `run.py` per integrazione
4. Creare config JSON per nested CV

Dimmi da dove iniziare! 🚀
