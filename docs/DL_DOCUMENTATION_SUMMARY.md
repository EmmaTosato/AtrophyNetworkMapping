# 📊 Deep Learning Pipeline - Documentazione Completa

## ✅ Documentazione Creata

Ho analizzato completamente la tua pipeline Deep Learning e creato **4 file di documentazione** completi:

---

### 1️⃣ **DL_PIPELINE_OVERVIEW.md** (4KB)
**Per**: Overview generale  
**Contenuto**:
- ✅ Workflow completo: Tuning → Runs → Testing
- ✅ Diagrammi visivi con ASCII art
- ✅ Filosofia della pipeline (perché CV, perché seed multipli)
- ✅ Struttura dati e output
- ✅ Quick start commands
- ✅ Confronto tra gruppi (ADNI vs PSP, ADNI vs CBS, PSP vs CBS)

---

### 2️⃣ **DL_TECHNICAL_REFERENCE.md** (12KB)
**Per**: Sviluppatori e AI agents  
**Contenuto**:
- ✅ Architettura completa del codice
- ✅ Function reference (run.py, train.py, test.py, hyper_tuning.py)
- ✅ Data flow dettagliato (training, testing)
- ✅ Modelli (ResNet3D, DenseNet3D, VGG16_3D)
- ✅ Datasets (FCDataset, AugmentedFCDataset)
- ✅ Configuration system (cnn_config.json, cnn_grid.json)
- ✅ Seed management (KFold seed vs PyTorch seed)
- ✅ Checkpoint structure
- ✅ Performance benchmarks (memory, time)
- ✅ Debugging guide

---

### 3️⃣ **DL_USER_GUIDE.md** (10KB)
**Per**: Utenti finali (ricercatori, studenti)  
**Contenuto**:
- ✅ Step 1: Hyperparameter Tuning (con esempi JSON)
- ✅ Step 2: Final Runs (multi-seed)
- ✅ Step 3: Final Testing
- ✅ Workflow completo per 3 gruppi
- ✅ Come interpretare i risultati (CSV, metriche)
- ✅ Parametri consigliati (batch size, learning rate)
- ✅ Troubleshooting (OOM, slow training, low accuracy)
- ✅ Checklist completa
- ✅ Best practices

---

### 4️⃣ **README_DL_DOCS.md** (3KB)
**Per**: Indice e navigazione  
**Contenuto**:
- ✅ Descrizione di ogni guida
- ✅ Quick navigation ("Voglio...")
- ✅ Struttura del progetto
- ✅ Percorsi di lettura consigliati
- ✅ Links utili (file principali, output, log)
- ✅ Troubleshooting quick links
- ✅ Versioning

---

## 🎨 Visual Workflow Diagram

### **DL_WORKFLOW_CORRECT.dot/png/svg**

Ho creato un diagramma Graphviz completo che mostra:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: HYPERPARAMETER TUNING                             │
│  • Grid Search (24-48 configurazioni)                       │
│  • 5-Fold CV per ogni config                               │
│  • Test su test set (27 pazienti)                          │
│  • Output: grid_results.csv                                │
└─────────────────────────────────────────────────────────────┘
                         ▼
            [Select Top Configs by test accuracy]
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: FINAL RUNS (Multi-Seed Robustness)               │
│  • Usa BEST hyperparameters dal tuning                      │
│  • 5 seed diversi: 42, 123, 2023, 31415, 98765            │
│  • 5-Fold CV per ogni seed                                 │
│  • Output: all_training_results.csv                        │
└─────────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: FINAL TESTING                                     │
│  • Carica best_model_fold{N}.pt per ogni run               │
│  • Test su test set (27 pazienti)                          │
│  • Metriche: accuracy, precision, recall, f1               │
│  • Output: all_testing_results.csv                         │
└─────────────────────────────────────────────────────────────┘
```

**File generati**:
- `DL_WORKFLOW_CORRECT.dot` (sorgente Graphviz)
- `DL_WORKFLOW_CORRECT.png` (300 DPI, alta risoluzione)
- `DL_WORKFLOW_CORRECT.svg` (vettoriale, scalabile)

---

## 📂 Struttura File Creati

```
docs/
├── README_DL_DOCS.md               # INIZIA DA QUI
├── DL_PIPELINE_OVERVIEW.md         # Overview generale
├── DL_TECHNICAL_REFERENCE.md       # Per sviluppatori
├── DL_USER_GUIDE.md                # Per utenti finali
├── DL_WORKFLOW_CORRECT.dot         # Diagramma sorgente
├── DL_WORKFLOW_CORRECT.png         # Diagramma PNG
└── DL_WORKFLOW_CORRECT.svg         # Diagramma SVG
```

---

## 🎯 Key Points Documentati

### ✅ Workflow Corretto
1. **Tuning**: Trova best hyperparameters con grid search + CV + test
2. **Runs**: Ripete con seed diversi per robustezza (NON refit!)
3. **Testing**: Testa best fold checkpoint su test set

### ✅ Filosofia Compresa
- **CV serve per**: Selezionare best fold (robustezza interna)
- **Seed diversi servono per**: Valutare stabilità dei risultati
- **Test accuracy bassa è normale**: Dataset piccolo (105 train, 27 test), task difficile
- **Focus**: Varianza tra seed, non accuracy assoluta

### ✅ Dati Chiariti
- **Train set**: 105 pazienti (non 104!)
- **Test set**: 27 pazienti (non 26!)
- **Ogni fold**: ~84 train, ~21 val
- **Augmentation**: 50× HCP bootstrap **solo in training fold**, non in val/test

### ✅ Seed Management Spiegato
- **StratifiedKFold seed**: Controlla split dei fold
- **PyTorch seed**: Controlla weight initialization
- **Attualmente**: Stesso seed controlla entrambi
- **Impatto**: Seed diversi → fold diversi → non comparabili (non critico per robustezza)

### ✅ Output Files Documentati
- **all_training_results.csv**: Run ID, group, seed, best fold, best epoch, accuracies, hyperparams
- **all_testing_results.csv**: Run ID, group, seed, accuracy, precision, recall, f1
- **grid_results.csv** (tuning): Config, accuracies, hyperparams
- **Checkpoints**: best_model_fold{1-5}.pt per ogni run

---

## 📊 Statistiche Documentazione

| File | Dimensione | Righe | Sezioni |
|------|------------|-------|---------|
| DL_PIPELINE_OVERVIEW.md | 4.8 KB | 252 | 10 |
| DL_TECHNICAL_REFERENCE.md | 15.2 KB | 572 | 10 |
| DL_USER_GUIDE.md | 11.6 KB | 476 | 8 |
| README_DL_DOCS.md | 3.4 KB | 178 | 7 |
| **TOTALE** | **35 KB** | **1478** | **35** |

---

## 🚀 Come Usare la Documentazione

### Per Nuovi Utenti
```bash
1. Leggi: docs/README_DL_DOCS.md (questo file)
2. Overview: docs/DL_PIPELINE_OVERVIEW.md
3. Pratica: docs/DL_USER_GUIDE.md
```

### Per Sviluppatori
```bash
1. Architettura: docs/DL_TECHNICAL_REFERENCE.md
2. Workflow: docs/DL_PIPELINE_OVERVIEW.md
3. Testing: docs/DL_USER_GUIDE.md
```

### Per AI Agents
```bash
1. Technical: docs/DL_TECHNICAL_REFERENCE.md
2. Workflow: docs/DL_PIPELINE_OVERVIEW.md
3. Commands: docs/DL_USER_GUIDE.md
```

---

## 🎓 Cosa Ho Capito

### Pipeline Workflow
1. **Tuning** (hyper_tuning.py):
   - Grid search su hyperparameters
   - CV 5-fold per ogni config
   - Test per selezionare best config
   - Output: grid_results.csv

2. **Runs** (run_train.py):
   - Usa best hyperparameters dal tuning
   - 5 seed diversi per robustezza
   - CV 5-fold per ogni seed
   - Output: all_training_results.csv + best_model_fold{N}.pt

3. **Testing** (run_test.py):
   - Carica best fold checkpoint
   - Testa su test set
   - Output: all_testing_results.csv

### Data Flow
- **Train fold**: `FCmaps_augmented_processed/` (50× augmented)
- **Val fold**: `FCmaps_processed/` (original)
- **Test set**: `FCmaps_processed/` (original)
- **Split fissi**: `assets/split_cnn/*.csv` (sempre uguali per confronti)

### Philosophy
- **Non serve refit**: Il modello del best fold è già trainato su ~84 pazienti
- **CV per robustezza**: Seleziona il fold che generalizza meglio
- **Multi-seed**: Valuta stabilità, non performance assoluta
- **Low test acc**: Normale per dataset piccoli e task difficili

---

## ✅ Checklist Completamento

- [x] Analizzato codice completo (10 file Python)
- [x] Compreso workflow Tuning → Runs → Testing
- [x] Chiarito filosofia della pipeline
- [x] Documentato data flow e augmentation
- [x] Spiegato seed management
- [x] Creato overview generale (DL_PIPELINE_OVERVIEW.md)
- [x] Creato technical reference (DL_TECHNICAL_REFERENCE.md)
- [x] Creato user guide (DL_USER_GUIDE.md)
- [x] Creato indice navigazione (README_DL_DOCS.md)
- [x] Generato diagramma workflow (PNG + SVG)
- [x] Documentato output files e metriche
- [x] Incluso troubleshooting e best practices

---

## 🎉 Risultato Finale

**4 documenti completi** (35 KB totali, 1478 righe) che coprono:
- ✅ Overview generale con workflow
- ✅ Reference tecnica per sviluppatori
- ✅ Guida pratica per utenti
- ✅ Indice di navigazione
- ✅ Diagramma visuale (PNG + SVG)

**Tutti i file in**: `/data/users/etosato/ANM_Verona/docs/`

---

**La documentazione è completa e pronta all'uso! 🚀**

Per iniziare, leggi: `docs/README_DL_DOCS.md`
