import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import umap

# Add src to path to import local modules
sys.path.append("src")
from ML_analysis.loading.config import ConfigLoader
from ML_analysis.analysis.classification import DataSplit
from ML_analysis.ml_utils import run_umap, log_to_file, reset_stdout

def get_best_params(dataset, model_name, group1, group2, base_results_dir="results/ML"):
    """
    Locates the nested_cv_all_results.csv for the specific experiment
    and extracts the parameters of the best performing fold.
    """
    # Construct path to results
    # Logic matches generate_report.py lookup
    # e.g. results/ML/voxel/umap_classification/ad_psp/csv_results/nested_cv_all_results.csv
    
    # Identify method folder name
    if "voxel" in dataset:
        ds_folder = "voxel"
        # We assume UMAP was used for voxel as per standard pipeline config, but check logic if needed.
        # For simplicity in this script, we check widely.
        # Actually, let's assume the standard path structure.
        method_folder = "umap_classification" 
    else:
        ds_folder = "networks"
        method_folder = "classification"
        
    # pair_name = f"{group1.lower()}_{group2.lower()}"
    # possible_pairs = [pair_name, f"{group2.lower()}_{group1.lower()}"]
    
    # Updated: Check Uppercase (AD_PSP) and Original Casing
    # The folders detected are AD_CBS, AD_PSP etc.
    possible_pairs = [
        f"{group1}_{group2}",
        f"{group2}_{group1}",
        f"{group1.lower()}_{group2.lower()}",
        f"{group2.lower()}_{group1.lower()}"
    ]
    
    csv_path = None
    for p in possible_pairs:
        # Correct path: group/model/nested_cv_results.csv
        path = os.path.join(base_results_dir, ds_folder, method_folder, p, model_name, "nested_cv_results.csv")
        if os.path.exists(path):
            csv_path = path
            break
            
    if not csv_path:
        print(f"[WARN] Could not find results CSV for {dataset} {group1}vs{group2} model {model_name}")
        return None

    try:
        df = pd.read_csv(csv_path)
        # The file is specific to the model, so no need to filter by model name column if it doesn't exist.
        # But let's check if 'model' column exists just in case, or just take the best row.
        if "model" in df.columns:
            df_model = df[df["model"] == model_name]
        else:
            df_model = df # Assume all rows are for this model
            
        if df_model.empty:
            return None
        # Sort by accuracy descending
        best_row = df_model.sort_values(by="accuracy", ascending=False).iloc[0]
        params = ast.literal_eval(best_row["best_params"])
        return params
    except Exception as e:
        print(f"[ERR] Failed parsing {csv_path}: {e}")
        return None

def get_model_instance(model_name, params, seed):
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=seed, **params)
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_permutation_analysis(config_path, suffix=None):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(">>> STARTING PERMUTATION ANALYSIS <<<")
    
    loader = ConfigLoader()
    # Base load (we will override filters dynamically)
    base_args, df_full, meta = loader.load_all()
    
    for comp in config["comparisons"]:
        group1, group2 = comp["groups"]
        dataset_type = comp["dataset"]
        use_umap = comp["umap"]
        
        print(f"\n--- Processing {dataset_type} | {group1} vs {group2} ---")
        
        # 1. Resolve Data Path specific to dataset_type (voxel vs networks)
        # We need to re-resolve because ConfigLoader loads what's in ml_config.json default.
        # We assume loader helper methods are available.
        # Hack: modify loader.args temporarily
        loader.args["dataset_type"] = dataset_type
        # If thresholding logic exists in resolve_data_path, ensure it's respected
        # Simplified: We trust ConfigLoader to have a helper or we manually resolve.
        # Let's inspect ConfigLoader.load_all implementation... it uses self.args.
        # Re-loading to be safe
        loader.args["dataset_type"] = dataset_type
        # Re-trigger load might be needed if resolving path happens at init or load_all start.
        # To be robust, we rely on the fact that we can filter the DF if it was all loaded, 
        # BUT voxel and networks are different files!
        # We must re-call `loader._resolve_data_path` logic.
        
        df_path = loader._resolve_data_path(
             dataset_type=dataset_type,
             threshold=loader.args.get("threshold", False),
             flat_args=loader.args
        )
        
        # Manually load the specific dataframe for this iteration
        # This duplicates some logic from loading.py but ensures correctness.
        from ML_analysis.loading.loading import load_metadata # Reuse existing function?
        # Actually easier: Just use pd.read_csv / load_func based on extension
        if df_path.endswith(".nii") or df_path.endswith(".nii.gz"):
             # It's an atlas/map, logic is complex.
             # Better to re-instantiate ConfigLoader with overridden args?
             # Or trust that `loader.load_all` does look at `loader.args["dataset_type"]`
             pass
        
        # Re-instantiate loader with override args would be cleaner but ConfigLoader loading from json...
        # Let's override the internal args dict before calling load_csv/etc.
        # Actually simplest way: Just use the previously implemented logic in `run_permutation.py` (old)
        # which called:
        # loader.args["dataset_type"] = ...
        # loader.args["df_path"] = ...
        loader.args["df_path"] = df_path
        
        # Now load
        _, df_current, _ = loader.load_all()
        
        # Filter Groups
        df_filtered = df_current[df_current["Group"].isin([group1, group2])].copy()
        
        # DataPrep
        # Use DataSplit to handle encoding and dropping metadata
        data_container = DataSplit(df_filtered, split_path=None, use_full_input=False)
        data_container.prepare_features()
        X = data_container.x_all
        y = data_container.y_encoded
        
        # UMAP (Global) - REMOVED to avoid leakage
        # if use_umap:
        #     print(f"Applying UMAP (Shape before: {X.shape})")
        #     X = run_umap(X, n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean', random_state=42)
        #     print(f"UMAP applied (Shape after: {X.shape})")
            
        # Iterate Models
        for model_name in config["models"]:
            # Retrieve Best Params
            best_params = get_best_params(dataset_type, model_name, group1, group2)
            
            if not best_params:
                print(f"[SKIP] No best params found for {model_name}")
                continue
                
            # Output Directory
            # Output Directory
            # Configured to save in the Pair Folder (e.g., .../AD_PSP/permutation_stats.csv)
            # This aggregates all models into one file
            results_dir = f"results/ML/{dataset_type}/{'umap_' if use_umap else ''}classification/{group1}_{group2}"
            
            # Define filename early
            filename = "permutation_stats.csv"
            if suffix:
                filename = f"permutation_stats{suffix}.csv"

            # Check if likely already done (naive check by reading file?)
            # Since we append, we run the risk of duplicates if we don't check content.
            # Let's read the file if it exists to check for existing model entry?
            full_out_path = os.path.join(results_dir, filename)
            if os.path.exists(full_out_path):
                 try:
                     existing_df = pd.read_csv(full_out_path)
                     if not existing_df.empty and model_name in existing_df["model"].values:
                          print(f"  > {model_name}: Already done in {filename}.")
                          continue
                 except:
                     pass # If error reading, assume we need to run or overwrite eventually
            
            print(f"  > Running {model_name} (Params: {best_params})...")
            
            # Setup Model
            base_model = get_model_instance(model_name, best_params, config["seed"])
            
            if use_umap:
                # Use Pipeline to run UMAP *inside* CV folds
                model = Pipeline([
                    ('umap', umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean', random_state=42)),
                    ('clf', base_model)
                ])
            else:
                model = base_model

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Run Test
            score, perm_scores, pvalue = permutation_test_score(
                estimator=model,
                X=X,
                y=y,
                scoring="accuracy",
                cv=cv,
                n_permutations=config["n_permutations"],
                n_jobs=config["n_jobs"],
                verbose=0
            )
            
            # Save
            os.makedirs(results_dir, exist_ok=True)
            res = {
                "dataset": dataset_type,
                "groups": f"{group1}vs{group2}",
                "model": model_name,
                "n_perms": config["n_permutations"],
                "true_score": score,
                "p_value": pvalue,
                "perm_scores_mean": np.mean(perm_scores),
                "perm_scores_std": np.std(perm_scores)
            }
            
            df_res = pd.DataFrame([res])
            if os.path.exists(full_out_path):
                df_res.to_csv(full_out_path, mode="a", header=False, index=False)
            else:
                df_res.to_csv(full_out_path, mode="w", header=True, index=False)
                
            print(f"    Saved to {filename}. p-value: {pvalue:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/ML_analysis/config/permutation_config.json")
    parser.add_argument("--suffix", help="Suffix for output file (e.g. '1000')")
    args = parser.parse_args()
    
    run_permutation_analysis(args.config, args.suffix)
