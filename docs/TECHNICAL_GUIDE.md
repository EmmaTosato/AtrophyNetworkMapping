# ML Classification Pipeline - Technical Documentation

## Architecture

```
ConfigLoader → DataSplit → [UMAP] → Model Training → Evaluation → Results
```

## Implementation Details

### 1. Classification Pipeline (`src/analysis/classification.py`)

Main function: `train_and_evaluate_model()`

```python
if params["tuning"]:
    # GridSearchCV on training set
    grid = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(5))
    grid.fit(data.x_train, data.y_train)
    best_model = grid.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(data.x_test)
    metrics = evaluate_metrics(data.y_test, y_pred)
else:
    # Train with fixed parameters
    model.set_params(**param_dict)
    model.fit(data.x_train, data.y_train)
    y_pred = model.predict(data.x_test)
    metrics = evaluate_metrics(data.y_test, y_pred)

# Permutation test on training set
if params["permutation_test"]:
    run_permutation_test(X=data.x_train, y=data.y_train, cv=5)
```

### 2. UMAP Implementation (`src/utils/ml_utils.py`)

```python
def run_umap(x_train, x_test=None):
    reducer = umap.UMAP(n_components=2)
    x_train_embedded = reducer.fit_transform(x_train)
    
    if x_test is not None:
        x_test_embedded = reducer.transform(x_test)
        return x_train_embedded, x_test_embedded
    
    return x_train_embedded
```

Key: fit only on training, transform test set separately

---

## Critical Issues Fixed

### Issue 1: Data Leakage (umap_all=True)

**Problem**: UMAP fitted on train+test concatenated data

```python
# WRONG
x_all = np.concatenate([x_train, x_test])
x_all_umap = run_umap(x_all)
```

**Solution**: Removed umap_all parameter. UMAP now fits only on training set.

### Issue 2: No Test Evaluation with Tuning

**Problem**: GridSearchCV did not evaluate on test set

**Solution**: Added test set evaluation after grid.fit()

### Issue 3: Permutation Test on Test Set

**Problem**: Test set used twice (evaluation + permutation)

**Solution**: Permutation test now runs on training set with cross-validation

### Issue 4: Conditional Output Saving

**Problem**: Results only saved when tuning=False

**Solution**: Results always saved regardless of tuning mode

---

## Recommended Improvements

### High Priority

1. **Cross-validation strategy** ✅ IMPLEMENTED
   - Old: Single train/test split
   - New: Optional nested CV with `use_fixed_split=false`
   - Benefit: Robust performance estimates with mean ± std

2. **Feature selection**
   - Current: All voxels/features used
   - Issue: High dimensionality, potential overfitting
   - Recommendation: Add feature selection step (e.g., ANOVA F-test, recursive feature elimination)

3. **Model evaluation metrics**
   - Current: Accuracy, precision, recall, F1
   - Issue: Class imbalance not addressed
   - Recommendation: Add balanced accuracy, Cohen's kappa, per-class metrics

4. **Hyperparameter optimization**
   - Current: Grid search with limited ranges
   - Issue: May miss optimal configurations
   - Recommendation: Use RandomizedSearchCV or Bayesian optimization

### Medium Priority

5. **Data validation**
   - Missing: Train/test overlap checks, NaN detection
   - Recommendation: Add validation function at pipeline start

6. **UMAP parameters**
   - Current: Fixed n_neighbors=15, min_dist=0.1
   - Issue: May not be optimal for data
   - Recommendation: Tune UMAP hyperparameters

7. **Model interpretability**
   - Missing: Feature importance, SHAP values
   - Recommendation: Add interpretation module

8. **Statistical power**
   - Current: Small test sets (n~20-40)
   - Issue: Limited generalization confidence
   - Recommendation: Report confidence intervals, power analysis

### Low Priority

9. **Code organization**
   - Current: Large monolithic functions
   - Recommendation: Refactor into smaller modules

10. **Documentation**
    - Current: Limited inline comments
    - Recommendation: Add docstrings, type hints

---

## Methodological Issues

### Small Sample Size
- Test sets: 20-40 subjects per comparison
- Impact: High variance in metrics, limited generalization
- Solution: Report confidence intervals, use bootstrap

### Class Imbalance
- Example: CN=58, AD=72, CBS=44, PSP=60
- Impact: Model may favor majority class
- Solution: Use class weights, stratified sampling, balanced accuracy

### Multiple Comparisons
- 3 binary comparisons × 3 models × 5 seeds = 45 tests
- Impact: Inflated Type I error
- Solution: Apply Bonferroni or FDR correction

### Fixed Train/Test Split
- Single split per comparison
- Impact: Results specific to that split
- Solution: Nested CV or repeated holdout
