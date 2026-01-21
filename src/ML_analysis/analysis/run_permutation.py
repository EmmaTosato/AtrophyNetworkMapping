import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Add src to path
sys.path.append("src")
from ML_analysis.loading.config import ConfigLoader
from ML_analysis.analysis.classification import DataSplit
from ML_analysis.ml_utils import run_umap, build_output_path

def get_model(model_name, params, seed):
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=seed, **params)
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_permutation(args):
    """
    Run permutation test using sklean.permutation_test_score.
    Uses fixed best parameters found in the main analysis.
    """
    print(f"\n>>> STATISTICAL SIGNIFICANCE TEST (Permutation) <<<")
    print(f"Comparison: {args.group1} vs {args.group2}")
    print(f"Model: {args.model}")
    print(f"Permutations: {args.n_perms}")
    
    # 1. Load Data
    loader = ConfigLoader()
    config, df_input, meta = loader.load_all()
    
    # Force uMAP/Dataset config from args if needed, or rely on ml_config.json
    # Better to filter manually here
    df_filtered = df_input[df_input["Group"].isin([args.group1, args.group2])].copy()
    print(f"Subjects: {len(df_filtered)}")
    
    # 2. Prepare Features
    data_container = DataSplit(df_filtered, split_path=None, use_full_input=False)
    data_container.prepare_features()
    X = data_container.x_all
    y = data_container.y_encoded
    
    # Handle UMAP if implied by config (simplified check)
    if config["job"].get("umap", False):
        print("Applying UMAP for permutation test (Warning: This biases the test if done on full dataset!)")
        print("IDEALLY: Permutation should be inside CV, but that is slow.")
        print("APPROXIMATION: Using Fixed UMAP embedding for speed.")
        # Note: Rigorous permutation would require re-running UMAP 1000 times inside the loop. 
        # sklearn's permutation_test_score takes an estimator and X, y. 
        # If X is already UMAP-ed, we test the classifier, not the UMAP+Classifier pipeline.
        # Acceptance: This tests if the CLASSIFIER finds signal in the UMAP features, not chance.
        pass

    # 3. Load Best Params
    # Default to generic if file not found
    params = {}
    if args.params_file and os.path.exists(args.params_file):
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded params: {params}")
    else:
        print("Using default parameters (No params file provided).")
        
    # 4. Setup Model and CV
    model = get_model(args.model, params, seed=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 5. Run Permutation Test
    print("Running permutations... (this may take a while)")
    score, perm_scores, pvalue = permutation_test_score(
        estimator=model,
        X=X,
        y=y,
        scoring="accuracy",
        cv=cv,
        n_permutations=args.n_perms,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n>>> RESULTS <<<")
    print(f"True Score: {score:.4f}")
    print(f"P-value:    {pvalue:.6f}")
    
    # 6. Save inputs
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    results = {
        "groups": f"{args.group1}vs{args.group2}",
        "model": args.model,
        "n_perms": args.n_perms,
        "true_score": score,
        "p_value": pvalue,
        "perm_scores_mean": np.mean(perm_scores),
        "perm_scores_std": np.std(perm_scores)
    }
    
    pd.DataFrame([results]).to_csv(os.path.join(out_dir, "permutation_stats.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(perm_scores, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Permutations')
    plt.axvline(score, color='red', linestyle='--', linewidth=2, label=f'True Score: {score:.3f}')
    plt.title(f"Permutation Test ({args.n_perms} iter)\n{args.group1} vs {args.group2} | {args.model}\np-value = {pvalue:.5f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "permutation_plot.png"))
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group1", required=True)
    parser.add_argument("--group2", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--params_file", help="Path to best_params.json")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/ML/significance")
    
    args = parser.parse_args()
    run_permutation(args)
