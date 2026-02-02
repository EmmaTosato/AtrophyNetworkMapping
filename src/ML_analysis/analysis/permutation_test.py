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
from sklearn.preprocessing import StandardScaler

def get_best_params(dataset, model_name, group1, group2, base_results_dir="results/ML"):
    """
    Locates the nested_cv_all_results.csv for the specific experiment
    and extracts the parameters of the best performing fold.
    """
    # Identify method folder name
    if "voxel" in dataset:
        ds_folder = "voxel"
        method_folder = "umap_classification" 
    else:
        ds_folder = "networks"
        method_folder = "classification"
        
    possible_pairs = [
        f"{group1}_{group2}",
        f"{group2}_{group1}",
        f"{group1.lower()}_{group2.lower()}",
        f"{group2.lower()}_{group1.lower()}"
    ]
    
    csv_path = None
    for p in possible_pairs:
        path = os.path.join(base_results_dir, ds_folder, method_folder, p, model_name, "nested_cv_results.csv")
        if os.path.exists(path):
            csv_path = path
            break
            
    if not csv_path:
        print(f"[WARN] Could not find results CSV for {dataset} {group1}vs{group2} model {model_name}")
        return None

    try:
        df = pd.read_csv(csv_path)
        if "model" in df.columns:
            df_model = df[df["model"] == model_name]
        else:
            df_model = df 
            
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
    base_args, df_full, meta = loader.load_all()
    
    for comp in config["comparisons"]:
        group1, group2 = comp["groups"]
        dataset_type = comp["dataset"]
        use_umap = comp["umap"]
        
        print(f"\n--- Processing {dataset_type} | {group1} vs {group2} ---")
        
        # Resolve Data Path specific to dataset_type
        loader.args["dataset_type"] = dataset_type
        
        df_path = loader._resolve_data_path(
             dataset_type=dataset_type,
             threshold=loader.args.get("threshold", False),
             flat_args=loader.args
        )
        
        loader.args["df_path"] = df_path
        
        # Load data
        _, df_current, _ = loader.load_all()
        
        # Filter Groups
        df_filtered = df_current[df_current["Group"].isin([group1, group2])].copy()
        
        # DataPrep
        data_container = DataSplit(df_filtered, split_path=None, use_full_input=False)
        data_container.prepare_features()
        X = data_container.x_all
        y = data_container.y_encoded
            
        # Iterate Models
        for model_name in config["models"]:
            # Retrieve Best Params
            best_params = get_best_params(dataset_type, model_name, group1, group2)
            
            if not best_params:
                print(f"[SKIP] No best params found for {model_name}")
                continue
                
            # Output Directory
            base_out = loader.args.get("output_dir", "results/ML/")
            results_dir = f"{base_out}/{dataset_type}/{'umap_' if use_umap else ''}classification/{group1}_{group2}"
            
            filename = "permutation_stats.csv"
            if suffix:
                filename = f"permutation_stats{suffix}.csv"

            full_out_path = os.path.join(results_dir, filename)
            
            # Check if likely already done
            if os.path.exists(full_out_path):
                 try:
                     existing_df = pd.read_csv(full_out_path)
                     if not existing_df.empty and model_name in existing_df["model"].values:
                          print(f"  > {model_name}: Already done in {filename}.")
                          continue
                 except:
                     pass 
            
            print(f"  > Running {model_name} (Params: {best_params})...")
            
            # Setup Model
            base_model = get_model_instance(model_name, best_params, config["seed"])
            
            # Construct Pipeline: Scaler -> (UMAP) -> Classifier
            pipeline_steps = [('scaler', StandardScaler())]
            
            if use_umap:
                # UMAP inside CV to avoid data leakage
                pipeline_steps.append(('umap', umap.UMAP(
                    n_neighbors=15, 
                    n_components=2, 
                    min_dist=0.1, 
                    metric='euclidean', 
                    random_state=42
                )))
            
            pipeline_steps.append(('clf', base_model))
            model = Pipeline(pipeline_steps)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Run Test
            score, perm_scores, pvalue = permutation_test_score(
                estimator=model,
                X=X,
                y=y,
                scoring="accuracy",
                cv=cv,
                n_permutations=config.get("n_permutations", 1000),
                n_jobs=config.get("n_jobs", -1),
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
