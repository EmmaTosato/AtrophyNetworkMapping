# 🚀 Deep Learning Pipeline - User Guide

## Quick Start: From Tuning to Testing

Questa guida ti spiega come usare la pipeline Deep Learning in 3 semplici step.

---

## 📋 Prerequisiti

- **Dati pronti**: 
  - `data/FCmaps_processed/` → Dati originali
  - `data/FCmaps_augmented_processed/` → Dati augmented (50×)
- **Split fissi**: `assets/split_cnn/{ADNI_PSP, ADNI_CBS, PSP_CBS}_splitted.csv`
- **GPU**: Consigliata (training su CPU molto lento)

---

## 🎯 Step 1: Hyperparameter Tuning

### Cosa fa?
Trova i migliori hyperparameters testando diverse combinazioni di:
- Modelli (ResNet, DenseNet, VGG16)
- Learning rate (1e-3, 1e-4, 1e-5)
- Batch size (4, 8, 16)
- Optimizer (Adam, SGD)
- Weight decay (1e-4, 1e-3)

### Come configurare?

Edita `src/DL_analysis/config/cnn_grid.json`:
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
    "group1": "ADNI",      # ← Cambia gruppo qui
    "group2": "PSP",       # ← 
    "run_id": 1,           # ← ID del tuning
    "tuning_flag": true
  }
}
```

### Esegui il tuning

```bash
cd /data/users/etosato/ANM_Verona
python src/DL_analysis/training/hyper_tuning.py
```

**Tempo stimato**: 30-40 ore per ~24 configurazioni

### Output

```
results/tuning/tuning{N}/
├── config1/
│   ├── best_model_fold1.pt
│   ├── best_model_fold2.pt
│   ├── ...
│   └── best_model_fold5.pt
├── config2/
│   └── ...
└── grid_results.csv  ← Risultati di tutte le config
```

### Scegli le migliori config

Apri `results/tuning/tuning{N}/grid_results.csv` e ordina per `best_accuracy` decrescente.

**Esempio**:
```csv
config,group,best_accuracy,avg_accuracy,model_type,lr,batch_size,optimizer
config2,ADNI vs PSP,0.857,0.80,densenet,0.001,8,adam
config8,ADNI vs PSP,0.905,0.79,densenet,0.0001,8,adam
config6,ADNI vs PSP,0.857,0.76,densenet,0.0001,8,adam
```

Seleziona le top 3-5 config per la fase successiva.

---

## ⚙️ Step 2: Final Runs (Multi-Seed)

### Cosa fa?
Ripete il training con i migliori hyperparameters usando **5 seed diversi** per valutare la robustezza dei risultati.

### Come configurare?

Edita `src/DL_analysis/config/cnn_config.json`:
```json
{
  "training": {
    "model_type": "densenet",    # ← Usa best model dal tuning
    "epochs": 50,
    "batch_size": 8,             # ← Usa best batch size
    "lr": 0.001,                 # ← Usa best lr
    "weight_decay": 0.0001,      # ← Usa best weight_decay
    "optimizer": "adam",         # ← Usa best optimizer
    "n_folds": 5,
    "seed": 42                   # ← Seed iniziale (script lo varia)
  },
  "experiment": {
    "group1": "ADNI",            # ← Gruppo da testare
    "group2": "PSP",
    "crossval_flag": true,       # ← IMPORTANTE: True
    "evaluation_flag": false,    # ← IMPORTANTE: False (solo training)
    "tuning_flag": false
  }
}
```

### Esegui le runs

```bash
cd /data/users/etosato/ANM_Verona
python src/DL_analysis/training/run_train.py
```

Questo script:
1. Legge le config da `cnn_config.json`
2. Ripete il training con 5 seed: `[42, 123, 2023, 31415, 98765]`
3. Salva 5 run consecutive (es. run1-5 per ADNI vs PSP)

**Tempo stimato**: ~6-8 ore per 5 run (1.2h × 5)

### Output

```
results/runs/
├── run1/ (seed 42)
│   ├── best_model_fold1.pt
│   ├── ...
│   ├── best_model_fold5.pt
│   ├── training_folds.xlsx
│   └── log_train1
├── run2/ (seed 123)
│   └── ...
├── ...
├── run5/ (seed 98765)
│   └── ...
└── all_training_results.csv  ← Summary di tutti i run
```

### Verifica risultati training

Apri `results/runs/all_training_results.csv`:
```csv
run_id,group,seed_x,best fold,best epoch,best accuracy,avg_accuracy
run1,ADNI vs PSP,42,4,5,0.857,0.829
run2,ADNI vs PSP,123,2,10,0.810,0.790
run3,ADNI vs PSP,2023,1,15,0.857,0.819
```

**Cosa guardare**:
- `best accuracy`: Val accuracy del best fold (deve essere ~80-90%)
- `avg_accuracy`: Media su tutti i fold (indica stabilità)
- `best fold` e `best epoch`: Usati per il testing

---

## 🧪 Step 3: Final Testing

### Cosa fa?
Testa i modelli trainati sul **test set** (27 pazienti, mai visti in training).

### Configurazione

Nessuna configurazione necessaria! Lo script legge automaticamente da `all_training_results.csv`.

### Esegui il testing

```bash
cd /data/users/etosato/ANM_Verona
python src/DL_analysis/testing/run_test.py
```

Questo script:
1. Legge `all_training_results.csv`
2. Per ogni run, carica `best_model_fold{N}.pt`
3. Testa sul test set (27 pazienti)
4. Salva metriche in `all_testing_results.csv`

**Tempo stimato**: ~15 minuti per 5 run

### Output

```
results/runs/
├── run1/
│   ├── log_test1
│   └── densenet_ADNI_vs_PSP_conf_matrix.png
├── ...
└── all_testing_results.csv  ← Metriche finali
```

### Analizza i risultati

Apri `results/runs/all_testing_results.csv`:
```csv
run_id,group,seed,accuracy,precision,recall,f1
run1,ADNI vs PSP,42,0.519,0.471,0.667,0.552
run2,ADNI vs PSP,123,0.630,0.600,0.750,0.667
run3,ADNI vs PSP,2023,0.556,0.520,0.700,0.595
```

**Metriche chiave**:
- `accuracy`: Accuratezza globale sul test set
- `precision`: Precisione (pochi falsi positivi)
- `recall`: Recall (pochi falsi negativi)
- `f1`: Media armonica di precision e recall

**Calcola media e deviazione standard**:
```python
import pandas as pd

df = pd.read_csv("results/runs/all_testing_results.csv")
print(df.groupby("group")["accuracy"].agg(["mean", "std"]))
```

---

## 📊 Workflow Completo per 3 Gruppi

Per confrontare tutti e 3 i gruppi (ADNI vs PSP, ADNI vs CBS, PSP vs CBS):

### 1. Tuning per tutti i gruppi

```bash
# Tuning 1: ADNI vs PSP
# Edita cnn_grid.json: group1="ADNI", group2="PSP", run_id=1
python src/DL_analysis/training/hyper_tuning.py

# Tuning 2: ADNI vs CBS
# Edita cnn_grid.json: group1="ADNI", group2="CBS", run_id=2
python src/DL_analysis/training/hyper_tuning.py

# Tuning 3: PSP vs CBS
# Edita cnn_grid.json: group1="PSP", group2="CBS", run_id=3
python src/DL_analysis/training/hyper_tuning.py
```

### 2. Runs per tutti i gruppi

Per ogni gruppo, modifica `cnn_config.json` con i best hyperparameters e lancia:

```bash
# Run 1-5: ADNI vs PSP
python src/DL_analysis/training/run_train.py

# Run 6-10: ADNI vs CBS
# Edita cnn_config.json: group1="ADNI", group2="CBS"
python src/DL_analysis/training/run_train.py

# Run 11-15: PSP vs CBS
# Edita cnn_config.json: group1="PSP", group2="CBS"
python src/DL_analysis/training/run_train.py
```

### 3. Testing per tutti

```bash
python src/DL_analysis/testing/run_test.py
```

Questo testa automaticamente **tutte** le run presenti in `all_training_results.csv`.

---

## ⚙️ Parametri Consigliati

### GPU Memory vs Batch Size

| GPU | ResNet/DenseNet | VGG16 |
|-----|-----------------|-------|
| 8GB | batch_size=8 | batch_size=4 |
| 12GB | batch_size=16 | batch_size=8 |
| 16GB+ | batch_size=16 | batch_size=16 |

### Learning Rate

| Model | Learning Rate Consigliato |
|-------|---------------------------|
| ResNet3D | 1e-3 (Adam) / 1e-2 (SGD) |
| DenseNet3D | 1e-3 (Adam) / 1e-2 (SGD) |
| VGG16_3D | 1e-4 (Adam) / 1e-3 (SGD) |

### Epochs

- **Tuning**: 20-50 epoch (per esplorare)
- **Final runs**: 50 epoch (con early stopping)

---

## 🐛 Troubleshooting

### Problema: Out of Memory (OOM)

**Soluzione**: Riduci `batch_size` in `cnn_config.json`:
```json
"batch_size": 4  // invece di 8 o 16
```

### Problema: Training molto lento

**Soluzione**: Controlla se stai usando la GPU:
```python
import torch
print(torch.cuda.is_available())  # Deve essere True
```

### Problema: Test accuracy molto bassa (<40%)

**Normale per dataset piccoli**! 
- Dataset: 105 train, 27 test
- Task difficile (malattie neurodegenerative simili)
- Focus: Varianza tra seed, non accuracy assoluta

### Problema: Seed diversi danno risultati molto diversi

**Normale**! È il motivo per cui usiamo 5 seed:
- Calcola media e std per valutare robustezza
- Se std > 10%, il modello è instabile

---

## 📝 Checklist Completa

### Prima di iniziare
- [ ] Dati in `data/FCmaps_processed/`
- [ ] Dati augmented in `data/FCmaps_augmented_processed/`
- [ ] Split CSV in `assets/split_cnn/`
- [ ] GPU disponibile

### Tuning
- [ ] Configura `cnn_grid.json` (gruppo, hyperparams)
- [ ] Esegui `hyper_tuning.py`
- [ ] Apri `grid_results.csv`
- [ ] Seleziona top config (best_accuracy)

### Runs
- [ ] Configura `cnn_config.json` con best hyperparams
- [ ] Esegui `run_train.py`
- [ ] Verifica `all_training_results.csv`
- [ ] Controlla log per errori

### Testing
- [ ] Esegui `run_test.py`
- [ ] Apri `all_testing_results.csv`
- [ ] Calcola mean ± std per accuracy
- [ ] Visualizza confusion matrices

---

## 💡 Best Practices

1. **Start small**: Prima testa con poche config (2-3) per verificare che tutto funzioni
2. **Monitor logs**: Controlla i log durante il training per errori
3. **Backup checkpoints**: Salva periodicamente `results/` (checkpoint pesanti!)
4. **Use tmux/screen**: Training lunghi → usa sessioni persistenti
5. **Track experiments**: Annota i parametri che funzionano meglio

---

## 📞 Support

Per domande o problemi:
- Controlla `docs/DL_TECHNICAL_REFERENCE.md` per dettagli tecnici
- Visualizza log files in `results/runs/run{N}/log_*`
- Verifica configurazioni in `src/DL_analysis/config/`

---

**Buon training! 🚀**
