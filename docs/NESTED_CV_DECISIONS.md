# 🎯 Nested CV: Decisioni Chiave e Roadmap

## 📊 Riepilogo Analisi

### ✅ TUA PROPOSTA: CORRETTA AL 100%!

```
OUTER 5-Fold CV (split diversi ogni fold)
  └─> INNER 5-Fold CV (grid search hyperparams)
      └─> FULL RETRAIN (best params, train completo)
          └─> TEST (fold esterno)
```

Questa è la **Nested Cross-Validation** standard - metodologia gold standard per:
- ✅ Evitare bias nella valutazione
- ✅ Sfruttare tutti i dati
- ✅ Stimare performance generalizzata

---

## 🔑 Risposte alle Tue Domande

### Q: "Dove metto i dati augmentati?"

**Risposta**:
```
OUTER FOLD:
  ├─ Train (80%) 
  │   ├─ INNER FOLD:
  │   │   ├─ Inner Train → AUGMENTED (50×) ✅
  │   │   └─ Inner Val → ORIGINAL ❌
  │   │
  │   └─ FULL RETRAIN → AUGMENTED (50×) ✅
  │
  └─ Test (20%) → ORIGINAL ❌
```

**Regola**: Augmentation **SOLO** in training, **MAI** in validation/test

---

### Q: "Grid search dal loop esterno?"

**SÌ**, ma grid search è **parte dell'Inner CV**:

```python
# CORRETTO:
for outer_fold in range(5):  # OUTER LOOP
    X_train_outer, X_test_outer = split(outer_fold)
    
    # Grid search DENTRO outer loop, SU train outer
    inner_cv = StratifiedKFold(5)
    grid = GridSearchCV(model, params, cv=inner_cv)  # INNER CV
    grid.fit(X_train_outer)  # NO leakage!
    
    best_params = grid.best_params_
    
    # Full retrain e test...
```

---

### Q: "Posso mantenere l'organizzazione attuale?"

**SÌ!** Modifiche minime:

✅ **MANTIENI**:
- Struttura `src/DL_analysis/` e `src/ML_analysis/`
- Models (`ResNet3D`, `VGG16_3D`, nuovo `AlexNet3D`)
- Datasets (`FCDataset`, `AugmentedFCDataset`)
- Utils (`cnn_utils.py`, `ml_utils.py`)
- Config system (JSON)

🔄 **MODIFICA**:
- `run.py`: Aggiungi outer loop + inner CV
- `hyper_tuning.py`: Integra in nested CV (o usa come inner CV)
- Split system: Dynamic splits (non più CSV fissi)
- Results structure: Nuova cartella `nested_cv/`

---

## 🚧 PROBLEMI DA RISOLVERE

### 1. Split System: CSV Fissi vs Dynamic

**Problema**: CSV fissi (`ADNI_PSP_splitted.csv`) incompatibili con Nested CV

**Soluzioni**:

#### Opzione A: Solo Dynamic (Raccomandato)
```python
# Genera split al volo
outer_cv = StratifiedKFold(5, random_state=42)
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split()):
    # Split dinamico per questo fold
```

**Pro**: Pulito, flessibile  
**Contro**: Non comparabile con vecchi risultati

#### Opzione B: Ibrido (Backward Compatible)
```python
if params.get("use_nested_cv", True):
    # New: Dynamic splits
    outer_cv = StratifiedKFold(5)
else:
    # Old: Fixed split from CSV
    split_path = resolve_split_csv_path(...)
```

**Pro**: Mantiene compatibilità  
**Contro**: Codice più complesso

**RACCOMANDAZIONE**: Opzione B (ibrido) per non perdere vecchi risultati

---

### 2. Grid Search: Separato o Integrato?

**Problema**: `hyper_tuning.py` attuale fa grid search standalone

**Soluzioni**:

#### Opzione A: Integra in Nested CV
```python
# nested_cv_runner.py
for outer_fold in outer_cv:
    # Inner CV integrato
    best_params = inner_cv_grid_search(train_outer)
    # ...
```

**Pro**: Tutto in un posto  
**Contro**: Codice monolitico

#### Opzione B: Riutilizza hyper_tuning.py
```python
# nested_cv_runner.py
for outer_fold in outer_cv:
    # Chiama hyper_tuning esistente
    best_params = hyper_tuning.tuning_on_fold(train_outer)
    # ...
```

**Pro**: Riuso codice esistente  
**Contro**: Duplicazione logica

**RACCOMANDAZIONE**: Opzione A (integrato) per evitare duplicazioni

---

### 3. Modelli: Quali Prioritizzare?

**Da implementare**:
1. ✅ **AlexNet3D** (nuovo)
2. ✅ **ResNet18_3D** (aggiornamento da ResNet34)
3. ✅ **VGG16_3D** (mantieni)

**Ordine di implementazione**:
1. **AlexNet3D** (più semplice, riferimento letteratura chiaro)
2. **ResNet18** (modifica minima: `r3d_18` vs `r3d_34`)
3. Test comparativo

---

## 📋 ROADMAP IMPLEMENTAZIONE

### Week 1: Core Infrastructure

#### Day 1-2: Nested CV Runner
- [ ] Crea `src/DL_analysis/training/nested_cv_runner.py`
- [ ] Implementa outer loop (5-fold)
- [ ] Integra con config system
- [ ] Test su dataset ridotto

#### Day 3-4: Inner CV Integration
- [ ] Implementa `inner_cv_grid_search()` in `nested_cv_runner.py`
- [ ] Integra logica da `hyper_tuning.py`
- [ ] Add progress tracking

#### Day 5: Full Retrain Logic
- [ ] Implementa `train_full_model()` 
- [ ] Gestisci augmented datasets
- [ ] Save/load checkpoints

### Week 2: Models & Testing

#### Day 6-7: AlexNet3D
- [ ] Implementa `AlexNet3D` class in `models.py`
- [ ] Add hyperparameters da letteratura
- [ ] Unit test forward pass
- [ ] Memory profiling

#### Day 8: ResNet18 Update
- [ ] Modifica `ResNet3D` → usa `r3d_18`
- [ ] Update constraints in `is_valid_combo()`
- [ ] Test comparativo ResNet18 vs VGG16

#### Day 9-10: Integration Testing
- [ ] End-to-end test su 1 gruppo (ADNI vs PSP)
- [ ] Verifica tempi (stimato: ~2-3 giorni per 5 fold × grid search)
- [ ] Debug issues

#### Day 11-12: ML Alignment
- [ ] Applica stessa logica a `src/ML_analysis/`
- [ ] Test nested CV su ML
- [ ] Verifica comparabilità risultati

### Week 3: Documentation & Finalization

#### Day 13-14: Documentation
- [ ] User guide: `NESTED_CV_USER_GUIDE.md`
- [ ] Technical reference: `NESTED_CV_TECHNICAL.md`
- [ ] Update README principale

#### Day 15: Validation
- [ ] Run su tutti e 3 i gruppi
- [ ] Confronta con old results (se disponibili)
- [ ] Generate final report

---

## 🎯 OUTPUTS ATTESI

### File Structure

```
results/
├── nested_cv/
│   ├── ADNI_vs_PSP/
│   │   ├── fold_1/
│   │   │   ├── inner_cv_results.csv       # Grid search CV interno
│   │   │   ├── best_hyperparams.json      # Best params selezionati
│   │   │   ├── best_model_fold1.pt        # Model full retrain
│   │   │   ├── test_results.csv           # Metriche su test fold
│   │   │   ├── training_log.txt
│   │   │   └── plots/
│   │   │       ├── confusion_matrix.png
│   │   │       └── learning_curves.png
│   │   ├── fold_2/ ...
│   │   ├── fold_3/ ...
│   │   ├── fold_4/ ...
│   │   ├── fold_5/ ...
│   │   ├── summary_all_folds.csv          # Aggregato
│   │   └── mean_std_metrics.csv           # Mean ± Std finale
│   │
│   ├── ADNI_vs_CBS/ ...
│   └── PSP_vs_CBS/ ...
│
└── old_runs/  # Backup vecchi risultati (opzionale)
```

### CSV Outputs

#### `summary_all_folds.csv`
```csv
fold,model,lr,batch_size,weight_decay,optimizer,accuracy,precision,recall,f1
1,AlexNet3D,0.001,16,0.0005,adam,0.72,0.70,0.75,0.72
2,AlexNet3D,0.001,16,0.0005,adam,0.68,0.65,0.72,0.68
...
```

#### `mean_std_metrics.csv`
```csv
model,metric,mean,std
AlexNet3D,accuracy,0.70,0.05
AlexNet3D,precision,0.68,0.06
AlexNet3D,recall,0.73,0.04
AlexNet3D,f1,0.70,0.05
ResNet18_3D,accuracy,0.68,0.07
...
```

---

## ⏱️ TEMPI STIMATI

### Training Time per Gruppo

**Configurazione**:
- 5 outer folds
- Inner CV: 5 folds × ~20 configs grid search
- Full retrain: 1× per fold esterno

**Breakdown**:
```
1 outer fold:
  ├─ Inner CV (5 folds × 20 configs × 20 epochs) = ~40 hours
  └─ Full retrain (50 epochs) = ~1.5 hours
  Total per fold: ~42 hours

5 outer folds: ~210 hours = ~9 giorni GPU continuativa
```

**Ottimizzazioni possibili**:
1. ✂️ Riduci grid (20 → 10 configs): **-50%** tempo
2. ✂️ Early stopping inner CV (20 → 10 epochs): **-50%** tempo  
3. ✂️ Parallelize folds (2 GPU): **-50%** tempo

**Tempo ottimizzato**: ~2-3 giorni per gruppo

---

## 🚀 NEXT ACTIONS

### Priorità 1: Approva Design
- [ ] Review `NESTED_CV_DESIGN.md`
- [ ] Conferma architettura proposta
- [ ] Decidi: Backward compatibility (opzione ibrida) o solo nuovo?

### Priorità 2: Choose Implementation Path
**Opzione A: Aggressive (2 settimane)**
- Implementa tutto da zero
- Nessuna backward compatibility
- Risultati puliti ma incomparabili

**Opzione B: Conservative (3 settimane)**
- Mantieni vecchia pipeline funzionante
- Aggiungi nested CV come opzione
- Risultati comparabili

**Raccomandazione**: Opzione B (conservative)

### Priorità 3: First Prototype
Vuoi che crei:
1. `nested_cv_runner.py` skeleton
2. `AlexNet3D` implementation
3. Modified config JSON
4. Test script

---

## 💬 DOMANDE PER TE

1. **Backward Compatibility**: Vuoi mantenere possibilità di usare vecchia pipeline (CSV fissi)?
   - [ ] Sì (opzione ibrida)
   - [ ] No (solo nested CV)

2. **Grid Search Size**: Quante configurazioni per inner CV?
   - [ ] Full grid (~50 configs) - accurato ma lento
   - [ ] Medium grid (~20 configs) - bilanciato
   - [ ] Small grid (~10 configs) - veloce ma limitato

3. **Modelli Priority**: Quale implementare per primo?
   - [ ] AlexNet3D
   - [ ] ResNet18
   - [ ] Entrambi insieme

4. **Test First**: Vuoi test su dataset ridotto (20 soggetti) prima di full run?
   - [ ] Sì (raccomandato)
   - [ ] No (vai diretto)

5. **ML Alignment**: Applicare nested CV anche a ML contemporaneamente?
   - [ ] Sì (DL + ML insieme)
   - [ ] No (prima DL, poi ML)

---

**Fammi sapere le tue decisioni e posso iniziare l'implementazione!** 🎯
