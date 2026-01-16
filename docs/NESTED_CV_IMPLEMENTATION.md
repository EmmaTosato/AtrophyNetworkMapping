# Implementazione Nested Cross-Validation

## Cosa Cambia

Invece di usare lo split fisso da CSV, il codice farà:

```
Outer Loop (5-fold):
  Fold 1: Train su 80%, Test su 20%
  Fold 2: Train su 80%, Test su 20%
  ...
  Fold 5: Train su 80%, Test su 20%
  
  Per ogni fold:
    Inner Loop (GridSearchCV con 5-fold):
      Trova best hyperparameters
    
    Valuta best model su test fold
```

## Modifiche al Codice

### 1. Aggiungi parametro in ml_config.json

```json
{
  "classification": {
    "use_nested_cv": true,       // NEW: attiva nested CV
    "n_outer_folds": 5,          // NEW: numero fold esterni
    "tuning": false,
    "n_folds": 5,                // Inner folds (per GridSearchCV)
    ...
  }
}
```

### 2. Modifica main_classification()

**Vecchio approccio**:
```python
# Legge split da CSV
split_path = resolve_split_csv_path(...)
data = DataSplit(df_input, split_path)
data.apply_split()  # Split fisso

for seed in seeds:
    classification_pipeline(data, params)  # Usa sempre lo stesso train/test
```

**Nuovo approccio**:
```python
if params.get("use_nested_cv", False):
    # Nested CV: split diversi ogni fold
    nested_cv_classification(df_input, params)
else:
    # Vecchio metodo: split fisso da CSV
    split_path = resolve_split_csv_path(...)
    data = DataSplit(df_input, split_path)
    ...
```

### 3. Nuova funzione nested_cv_classification()

```python
def nested_cv_classification(df_input, params):
    """Run nested cross-validation with different splits."""
    from sklearn.model_selection import StratifiedKFold
    
    # Filtra solo i gruppi richiesti
    df = df_input[df_input["Group"].isin([params["group1"], params["group2"]])].copy()
    
    # Prepara feature e label
    meta_columns = ["ID", "Group", "Sex", "Age", "Education", "CDR_SB", "MMSE"]
    X = df.drop(columns=meta_columns).to_numpy()
    y = df["Group"].to_numpy()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Outer loop: 5 fold diversi
    outer_cv = StratifiedKFold(
        n_splits=params.get("n_outer_folds", 5),
        shuffle=True,
        random_state=42
    )
    
    all_results = []
    fold_idx = 0
    
    for train_idx, test_idx in outer_cv.split(X, y_encoded):
        fold_idx += 1
        print(f"\n{'='*50}")
        print(f"OUTER FOLD {fold_idx}/{params.get('n_outer_folds', 5)}")
        print(f"{'='*50}")
        
        # Split train/test per questo fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Applica UMAP se richiesto
        if params.get("umap", False):
            X_train, X_test = run_umap(X_train, X_test)
        
        # Crea oggetto DataSplit "manuale" per compatibilità
        data = DataSplit.__new__(DataSplit)
        data.x_train = X_train
        data.x_test = X_test
        data.y_train = y_train
        data.y_test = y_test
        data.le = le
        data.df = df
        data.splits = np.array(["train"] * len(train_idx) + ["test"] * len(test_idx))
        
        # Esegui classificazione su questo fold
        for seed in params["seeds"]:
            params["seed"] = seed
            params["fold"] = fold_idx
            set_seed(seed)
            
            df_summary, df_preds = classification_pipeline(data, params)
            
            if df_summary is not None:
                df_summary["outer_fold"] = fold_idx
                all_results.append(df_summary)
    
    # Aggregated results
    if all_results:
        df_all = pd.concat(all_results).reset_index(drop=True)
        
        # Media e std su tutti i fold
        summary = df_all.groupby("model").agg({
            "test_accuracy": ["mean", "std"],
            "test_precision": ["mean", "std"],
            "test_recall": ["mean", "std"],
            "test_f1": ["mean", "std"]
        }).round(3)
        
        print(f"\n{'='*50}")
        print("NESTED CV RESULTS - Aggregated across all folds and seeds")
        print(f"{'='*50}")
        print(summary)
        
        # Salva risultati
        output_dir = params["path_umap_classification"]
        df_all.to_csv(os.path.join(output_dir, "nested_cv_all_results.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "nested_cv_summary.csv"))
```

### 4. Modifica classification_pipeline()

Nessuna modifica necessaria! La funzione già lavora con un oggetto `DataSplit` che ha `x_train`, `x_test`, ecc.

## Come Usare

### Opzione A: Nested CV (Split Diversi)

**ml_config.json**:
```json
{
  "classification": {
    "use_nested_cv": true,
    "n_outer_folds": 5,
    "tuning": true,           // Inner GridSearchCV
    "n_folds": 3,             // Inner CV folds
    ...
  }
}
```

**Risultato**:
```
Outer Fold 1: accuracy = 70%
Outer Fold 2: accuracy = 65%
Outer Fold 3: accuracy = 60%
Outer Fold 4: accuracy = 68%
Outer Fold 5: accuracy = 63%

Mean accuracy: 65.2% ± 3.8%
```

### Opzione B: Single Split Fisso (Come Prima)

**ml_config.json**:
```json
{
  "classification": {
    "use_nested_cv": false,   // Usa CSV split
    "tuning": false,
    ...
  }
}
```

**Risultato**:
```
Single test set: accuracy = 65%
```

## Vantaggi Nested CV

1. **Robustezza**: Ogni paziente viene testato esattamente 1 volta
2. **Variabilità**: Ottieni media e deviazione standard
3. **Generalizzazione**: Risultati meno dipendenti da uno split specifico

## Svantaggi

1. **Tempo**: 5× più lento (5 outer folds)
2. **Bilanciamento**: Non puoi controllare manualmente train/test (fatto automaticamente con stratification)
3. **Confrontabilità**: Risultati diversi dal paper se usava split fisso

## Raccomandazione

- **Per analisi esplorative**: Usa nested CV
- **Per riportare risultati finali**: Se il paper usa split fisso matched per età/sesso, mantieni quello per confrontabilità
- **Per pubblicazione**: Riporta entrambi (split fisso + nested CV in supplementary)

---

Vuoi che implementi questa modifica nel codice?
