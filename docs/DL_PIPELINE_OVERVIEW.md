# 🧠 Deep Learning Pipeline - Overview

## 📊 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASE 1: HYPERPARAMETER TUNING                    │
└─────────────────────────────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Grid Search su combinazioni di:                 │
        │  • Model: ResNet3D, DenseNet3D, VGG16_3D        │
        │  • Optimizer: Adam, SGD                          │
        │  • Learning Rate: 1e-3, 1e-4, 1e-5             │
        │  • Batch Size: 4, 8, 16                         │
        │  • Weight Decay: 1e-4, 1e-3                     │
        └──────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Per ogni configurazione:                        │
        │  1. 5-Fold Cross-Validation su train (105 pz)   │
        │  2. Salva best_model_fold{1-5}.pt               │
        │  3. Seleziona best fold (val accuracy)           │
        │  4. Test su test set (27 pz)                     │
        └──────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Output:                                         │
        │  • grid_results.csv (tutte le config)            │
        │  • Seleziona migliori config (top accuracy)      │
        └──────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    FASE 2: FINAL RUNS (Robustness Check)                │
└─────────────────────────────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Usa i MIGLIORI hyperparameters dal tuning       │
        │  Ripeti con 5 seed diversi per robustezza:      │
        │  • Seed: 42, 123, 2023, 31415, 98765            │
        └──────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Per ogni seed (run1-15):                        │
        │  1. 5-Fold Cross-Validation su train (105 pz)   │
        │  2. Salva best_model_fold{1-5}.pt               │
        │  3. Seleziona best fold (val accuracy)           │
        └──────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Output:                                         │
        │  • all_training_results.csv                      │
        │  • best_model_fold{N}.pt per ogni run           │
        └──────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          FASE 3: FINAL TESTING                           │
└─────────────────────────────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Per ogni run:                                   │
        │  1. Carica best_model_fold{N}.pt                 │
        │  2. Testa su test set (27 pz)                    │
        │  3. Calcola metriche (acc, prec, rec, f1)        │
        └──────────────────────────────────────────────────┘
                                     ▼
        ┌──────────────────────────────────────────────────┐
        │  Output:                                         │
        │  • all_testing_results.csv                       │
        │  • Confusion matrices                            │
        │  • Performance report                            │
        └──────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### 1️⃣ **Data Splits** (Fissi per tutti gli esperimenti)
- **Train**: 105 pazienti (80%)
- **Test**: 27 pazienti (20%)
- File: `assets/split_cnn/{ADNI_PSP, ADNI_CBS, PSP_CBS}_splitted.csv`

### 2️⃣ **Data Augmentation** (Solo in Training)
- **Train fold**: `FCmaps_augmented_processed/` (50× augmented con HCP bootstrap)
- **Validation fold**: `FCmaps_processed/` (dati originali, no augmentation)
- **Test set**: `FCmaps_processed/` (dati originali, no augmentation)

### 3️⃣ **Cross-Validation** (5-fold Stratified)
- Ogni fold: ~84 pazienti train, ~21 pazienti validation
- **Best fold**: Scelto in base a validation accuracy
- **Best epoch**: Early stopping su validation loss

### 4️⃣ **Modelli Disponibili**
| Modello | Parametri | Batch Size Consigliato |
|---------|-----------|------------------------|
| ResNet3D | ~33M | 4, 16 |
| DenseNet3D | ~7M | 4, 16 |
| VGG16_3D | ~138M | 4, 8, 16 |

---

## 📂 Struttura Output

```
results/
├── tuning/
│   ├── tuning{N}/
│   │   ├── config{M}/
│   │   │   ├── best_model_fold{1-5}.pt
│   │   │   ├── plots/
│   │   │   └── training_folds.xlsx
│   │   └── grid_results.csv
│   ├── merged_{GROUP}.csv      # Risultati test per tuning
│   └── summary.csv             # Riassunto tutti i tuning
│
└── runs/
    ├── run{N}/
    │   ├── best_model_fold{1-5}.pt
    │   ├── plots/
    │   ├── training_folds.xlsx
    │   ├── log_train{N}
    │   └── log_test{N}
    ├── all_training_results.csv
    └── all_testing_results.csv
```

---

## 🎯 Philosophy: Tuning vs Runs

### **Tuning** (Fase 1)
- **Obiettivo**: Trovare i migliori hyperparameters
- **Metodo**: Grid search + CV + Test
- **Output**: Selezione delle config migliori

### **Runs** (Fase 2-3)
- **Obiettivo**: Valutare stabilità e robustezza
- **Metodo**: CV con seed diversi + Test
- **Output**: Metriche finali con varianza

⚠️ **Nota**: L'accuracy del test set è **bassa intenzionalmente** perché:
1. Dataset piccolo (105 train, 27 test)
2. Task difficile (discriminazione neurodegenerativa)
3. Focus su robustezza (varianza tra seed) non su performance assoluta

---

## 📈 Metriche Salvate

### Training (all_training_results.csv)
- `best_fold`: Fold selezionato
- `best_epoch`: Epoca di early stopping
- `best_accuracy`: Val accuracy del best fold
- `avg_accuracy`: Media val accuracy su tutti i fold
- Hyperparameters: model, lr, batch_size, optimizer, weight_decay

### Testing (all_testing_results.csv)
- `accuracy`: Accuracy su test set
- `precision`, `recall`, `f1`: Metriche dettagliate
- `seed`: Seed usato per il run

---

## 🚀 Quick Start

### 1. Hyperparameter Tuning
```bash
python src/DL_analysis/training/hyper_tuning.py
```

### 2. Final Runs
```bash
python src/DL_analysis/training/run_train.py
```

### 3. Final Testing
```bash
python src/DL_analysis/testing/run_test.py
```

---

## 📊 Confronto Gruppi

| Gruppo Pair | Train Size | Test Size | Tuning | Runs |
|-------------|------------|-----------|--------|------|
| ADNI vs PSP | 105 | 27 | tuning1-2 | run1-5 |
| ADNI vs CBS | 95 | 24 | tuning3 | run6-10 |
| PSP vs CBS | 80 | 20 | tuning4 | run11-15 |

---

## ⚙️ Configurazione

### File Principali
- `src/DL_analysis/config/cnn_config.json`: Config base (paths, hyperparams, flags)
- `src/DL_analysis/config/cnn_grid.json`: Grid search per tuning

### Flags Importanti
- `crossval_flag`: Abilita training con CV
- `evaluation_flag`: Abilita testing
- `tuning_flag`: Modalità tuning (ritorna risultati senza CSV)
- `plot`: Salva plot (learning curves, confusion matrices)
- `training_csv`: Salva metriche per-epoch in Excel

---

## 📝 Note Tecniche

### Seed Management
- **Seed nel CV**: Controlla split dei fold (stratified split)
- **Seed in PyTorch**: Controlla weight initialization
- Nelle runs: seed diversi per testare robustezza

### Early Stopping
- Basato su **validation accuracy** (primary)
- Tiebreaker: **validation loss** (secondary)
- Best epoch può essere molto precoce (es. epoch 5)

### GPU Memory
- Batch size dipende da modello e GPU disponibile
- VGG16: Max 8, ResNet/DenseNet: Max 16
- Validation constraints in `is_valid_combo()`

---

**Autore**: Pipeline DL per classificazione neurodegenerativa  
**Ultima modifica**: Gennaio 2026
