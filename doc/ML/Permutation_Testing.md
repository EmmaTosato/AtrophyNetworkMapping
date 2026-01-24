# Statistical Significance: Permutation Testing

This document provides a technical overview of the permutation testing methodology used to validate Machine Learning models in this project. Specifically, it details the implementation of `sklearn.model_selection.permutation_test_score`.

## 1. Theoretical Concept
Permutation testing validates whether a classifier has found a real signal in the data or if it is merely overfitting to noise. 

**The Null Hypothesis (H0):** 
> "The features contain no information about the class labels; the classifier performance is no better than random guessing."

To test this, we:
1.  **Break the relationship** between features (X) and labels (y) by randomly shuffling `y`.
2.  Train the model on this "meaningless" data and measure accuracy.
3.  Repeat this process $N$ times (e.g., 100 or 1000) to build a distribution of "chance" scores.
4.  Compare the **True Score** (from the real labels) against this distribution.

## 2. Methodology & Implementation

We utilize the standard `scikit-learn` implementation:
[sklearn.model_selection.permutation_test_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)

### Key Parameters used in this Project:
*   **Permutations ($N$)**: 100 (for speed) or 1000 (for final publication).
*   **Cross-Validation**: 5-Fold Stratified CV (`StratifiedKFold`).
*   **Scoring Metric**: Accuracy.
*   **Model**: The exact model instance with the **Best Hyperparameters** found during the Nested CV phase.

### Process Flow
1.  **Input**:
    *   Feature Matrix ($X$): Voxel-wise data (UMAP reduced) or Network Connectivity.
    *   True Labels ($y$): Diagnostic groups (e.g., AD vs PSP).
    *   Best Parameters: Loaded from the `nested_cv_all_results.csv` of the main analysis.
2.  **Execution**:
    *   Calculate **True Score**: Mean CV accuracy on $(X, y)$.
    *   Calculate **Permuted Scores**: Mean CV accuracy on $(X, y_{shuffled})$ for $N$ iterations.
3.  **P-Value Calculation**:
    The p-value represents the probability of observing a score as good as (or better than) the True Score by pure chance.
    $$ p = \frac{C + 1}{N + 1} $$
    Where:
    *   $C$: Number of permutations where $Score_{perm} \geq Score_{true}$.
    *   $N$: Total number of permutations.
    *   The $+1$ is a correction for the true score itself (conservative estimate).

## 3. Interpretation

### P-Value < 0.05
*   **Result**: Significant.
*   **Meaning**: The classifier's performance is extremely unlikely to be caused by chance. The model has learned a genuine pattern distinguishing the groups.

### P-Value >= 0.05
*   **Result**: Not Significant.
*   **Meaning**: We cannot reject the null hypothesis. The model's accuracy, even if seemingly high (e.g., 60%), is not statistically distinguishable from random noise variance in this specific dataset.

## 4. Output Files

For each model and comparison, the results are saved in the respective model directory (e.g., `results/ML/.../RandomForest/`):

*   **`permutation_plot.png`**: A histogram showing the distribution of chance scores (blue) vs the True Score (red line). 
    *   *Visual Check*: The red line should be far to the right of the blue distribution.
*   **`permutation_stats.csv`**: Contains the raw values (`true_score`, `p_value`, `n_perms`).
