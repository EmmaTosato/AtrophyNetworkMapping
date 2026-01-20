# Classification Modes Documentation

## Overview

Il codice supporta due modalità di classificazione controllate dal parametro `use_fixed_split` in `ml_config.json`.

---

## Mode 1: Fixed Split (use_fixed_split=true)

### Comportamento

```
Dataset (130 pazienti)
    ↓
Legge split da CSV (train/test fisso)
    ↓
Train: 104 pazienti (sempre gli stessi)
Test:  26 pazienti (sempre gli stessi)
    ↓
Ripete per 5 seeds (42, 123, 2023, 31415, 98765)
    ↓
Output: 5 risultati sullo stesso test set
```

### Configurazione

```json
{
  "classification": {
    "use_fixed_split": true,
    "seeds": [42, 123, 2023, 31415, 98765]
  }
}
```

### Output Files

```
results/ml_analysis/voxel/classification/psp_cbs/
├── seed_42/
│   ├── conf_matrix_test_RandomForest.png
│   ├── test_predictions_RandomForest.csv
│   └── ...
├── seed_123/
│   └── ...
├── summary_all_seeds.csv          # Metriche aggregate
└── all_test_predictions.csv       # Tutte le predizioni
```

### Risultati Attesi

**summary_all_seeds.csv**:
```
model             seed  test_accuracy  test_precision  test_recall  test_f1
RandomForest      42    0.653          0.645           0.651        0.648
RandomForest      123   0.650          0.642           0.648        0.645
RandomForest      2023  0.655          0.648           0.653        0.650
RandomForest      31415 0.651          0.643           0.649        0.646
RandomForest      98765 0.654          0.646           0.652        0.649
```

**Summary statistics** (stampato a console):
```
                 test_accuracy          test_precision
                 mean      std           mean      std
RandomForest     0.653    0.002         0.645    0.002
```

### Quando Usare

- Hai uno split predefinito matched per età/sesso
- Vuoi confrontare con risultati precedenti
- Vuoi un test set "vergine" mai visto
- Pubblichi risultati finali (comparabili)

### Pro/Contro

**Pro**:
- Controllo completo sullo split
- Test set mai usato per training
- Risultati confrontabili tra esperimenti

**Contro**:
- Risultati dipendono dallo split specifico
- Variabilità limitata (solo da inizializzazione random)

---

## Mode 2: Nested CV (use_fixed_split=false)

### Comportamento

```
Dataset (130 pazienti)
    ↓
Outer CV: 5 folds (StratifiedKFold)
    ↓
Fold 1: Train 104, Test 26 (subset A)
  → Ripete per 5 seeds → 5 risultati
Fold 2: Train 104, Test 26 (subset B)
  → Ripete per 5 seeds → 5 risultati
...
Fold 5: Train 104, Test 26 (subset E)
  → Ripete per 5 seeds → 5 risultati
    ↓
Output: 25 risultati totali (5 folds × 5 seeds)
```

### Configurazione

```json
{
  "classification": {
    "use_fixed_split": false,
    "n_outer_folds": 5,
    "seeds": [42, 123, 2023, 31415, 98765]
  }
}
```

### Output Files

```
results/ml_analysis/voxel/classification/psp_cbs/
├── fold_1/
│   ├── seed_42/
│   │   ├── conf_matrix_test_RandomForest.png
│   │   └── ...
│   ├── seed_123/
│   └── ...
├── fold_2/
│   └── ...
├── nested_cv_all_results.csv      # Tutti i 25 risultati
├── nested_cv_summary.csv          # Statistiche aggregate
└── nested_cv_all_predictions.csv  # Tutte le predizioni
```

### Risultati Attesi

**nested_cv_all_results.csv**:
```
model             outer_fold  seed  test_accuracy  test_precision  ...
RandomForest      1           42    0.670          0.665           ...
RandomForest      1           123   0.668          0.663           ...
RandomForest      2           42    0.620          0.615           ...
RandomForest      2           123   0.618          0.613           ...
...
(25 righe totali)
```

**nested_cv_summary.csv** (stampato a console):
```
Overall performance (all folds × all seeds):
                 test_accuracy          test_precision
                 mean      std           mean      std
RandomForest     0.648    0.035         0.642    0.033


Per-fold variability (averaged across seeds):

RandomForest:
  Fold accuracies: [0.669, 0.619, 0.655, 0.630, 0.665]
  Mean: 0.648 ± 0.023
```

### Quando Usare

- Dataset piccolo (<200 soggetti)
- Vuoi stimare variabilità dovuta allo split
- Analisi esplorative
- Vuoi report con intervalli di confidenza

### Pro/Contro

**Pro**:
- Uso efficiente dei dati (ogni paziente testato 1 volta)
- Stima robusta con media e std
- Indipendente da split specifico

**Contro**:
- 5× più lento
- Non controllabile lo split manualmente
- Risultati non direttamente confrontabili con split fisso

---

## Confronto Diretto

| Aspetto | Fixed Split | Nested CV |
|---------|-------------|-----------|
| **Split train/test** | Fisso da CSV | Variabile (5 fold) |
| **N. valutazioni** | 5 (seeds) | 25 (5 fold × 5 seeds) |
| **Test set** | Sempre 26 pazienti | 26 pazienti diversi per fold |
| **Variabilità misurata** | Inizializzazione random | Split + inizializzazione |
| **Tempo esecuzione** | 1× | 5× |
| **Output principale** | Single accuracy ± std(seeds) | Mean accuracy ± std(folds×seeds) |
| **Comparabilità** | Alta | Bassa |
| **Robustezza** | Moderata | Alta |

---

## Interpretazione Risultati

### Fixed Split

**Output**:
```
RandomForest: 0.653 ± 0.002 (across 5 seeds)
```

**Interpretazione**:
- L'accuracy media è 65.3%
- La deviazione standard è 0.2% (molto bassa)
- Significa: il modello è **stabile rispetto all'inizializzazione**
- **NON** significa: il modello funziona bene su qualsiasi split

**Come riportare**:
> "Using a fixed train/test split, RandomForest achieved 65.3% ± 0.2% accuracy (mean ± std across 5 random seeds)."

---

### Nested CV

**Output**:
```
RandomForest: 0.648 ± 0.035 (across 5 folds × 5 seeds)

Per-fold:
  Fold 1: 66.9%
  Fold 2: 61.9%
  Fold 3: 65.5%
  Fold 4: 63.0%
  Fold 5: 66.5%
```

**Interpretazione**:
- L'accuracy media è 64.8%
- La deviazione standard è 3.5% (moderata)
- Variabilità viene da:
  - **Fold diversi** (≈2.3% std)
  - **Seeds diversi** (≈0.2% std)
- Significa: il modello è **moderatamente robusto** a diversi split

**Come riportare**:
> "Using 5-fold stratified cross-validation repeated with 5 random seeds, RandomForest achieved 64.8% ± 3.5% accuracy (mean ± std across 25 evaluations)."

---

## Raccomandazioni

### Per Analisi Esplorative
```json
{
  "use_fixed_split": false,
  "n_outer_folds": 5,
  "seeds": [42, 123, 2023]  // 3 seeds sufficiente
}
```
→ Ottieni stima robusta rapidamente

---

### Per Risultati Finali (Paper)
```json
{
  "use_fixed_split": true,
  "seeds": [42, 123, 2023, 31415, 98765]
}
```
→ Usi split fisso comparabile, riporti anche nested CV in supplementary

---

### Per Dataset Piccoli (<100 soggetti)
```json
{
  "use_fixed_split": false,
  "n_outer_folds": 10,  // Leave-pair-out like
  "seeds": [42]
}
```
→ Massimizza uso dei dati

---

## Permutation Test

Il permutation test **funziona con entrambe le modalità**:

### Fixed Split
- Permutation test su **training set** con 5-fold CV
- P-value singolo (1 per seed)

### Nested CV
- Permutation test su **training set di ogni fold** con 5-fold CV
- P-value multipli (5 fold × 5 seeds = 25 p-values)
- Puoi fare media o riportare range

---

## Troubleshooting

### "Risultati troppo diversi tra fixed e nested CV"

**Normale**: Nested CV include variabilità dello split.

**Esempio**:
- Fixed: 65% ± 0.2%
- Nested: 64% ± 3.5%

Differenza di 1% è trascurabile, std maggiore è atteso.

---

### "Fold 2 ha accuracy molto bassa"

**Cause possibili**:
1. Per caso, quel fold ha pazienti difficili
2. Sbilanciamento classi nel fold (improbabile con StratifiedKFold)
3. Overfitting su fold specifico

**Verifica**:
```python
# Guarda distribuzione fold
print(df_all[df_all["outer_fold"] == 2].groupby("model")["test_accuracy"].describe())
```

---

### "Nested CV troppo lento"

**Soluzioni**:
1. Riduci `n_outer_folds` a 3
2. Riduci `seeds` a [42, 123]
3. Disabilita `permutation_test`
4. Usa meno modelli (solo RandomForest)

---

## Checklist Pre-Run

- [ ] Scelto `use_fixed_split` appropriato per il tuo caso
- [ ] Verificato `n_outer_folds` (default: 5)
- [ ] Verificato `seeds` (raccomandato: 5 seeds)
- [ ] Controllato `group1` e `group2` corretti
- [ ] Se nested CV: controllato che dataset abbia abbastanza soggetti (>50 per gruppo)
- [ ] Verificato spazio disco (nested CV crea molti file)

---

*Gennaio 2026*
