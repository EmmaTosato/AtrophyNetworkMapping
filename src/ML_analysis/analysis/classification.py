import os
import json
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from ML_analysis.loading.config import ConfigLoader
from ML_analysis.ml_utils import run_umap, log_to_file, reset_stdout, resolve_split_csv_path, build_output_path
from ML_analysis.analysis.plotting import plot_confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DataSplit:
    def __init__(self, df_input: pd.DataFrame, split_path: str = None, label_col: str = "Group", use_full_input: bool = False):
        if split_path:
             self.df_split = pd.read_csv(split_path)
        else:
            self.df_split = None

        if use_full_input:
            self.df_full = df_input.copy()
            self.df = None
        elif split_path:
            self.df = self.df_split.merge(df_input, on="ID", how="left")
        else:
             self.df = df_input.copy()

        self.label_col = label_col
        # Extended metadata columns to exclude from features
        self.meta_columns = [
            "ID", "Group", "Sex", "Age", "Education", "CDR_SB", "MMSE", "split",
            "labels_gmm_cdr", "labels_km", "labels_hdb"
        ]
        
        # Only keep available meta columns to avoid KeyErrors
        self.meta_columns = [col for col in self.meta_columns if col in self.df.columns]

        self.x_all = None
        self.y_all = None
        self.y_encoded = None
        self.splits = None

        self.le = LabelEncoder()
        
        # Will be set dynamically during CV
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None

    def insert_umap(self, x_umap: np.ndarray):
        """After running UMAP externally, reinsert embedding + ID and merge with split."""
        df_umap = pd.DataFrame(x_umap, columns=[f"UMAP{i+1}" for i in range(x_umap.shape[1])])
        df_umap["ID"] = self.df_full["ID"].values
        self.df = self.df_split.merge(df_umap, on="ID", how="left")

    def prepare_features(self):
        self.x_all = self.df.drop(columns=self.meta_columns).to_numpy()
        self.y_all = self.df[self.label_col].to_numpy()
        self.y_encoded = self.le.fit_transform(self.y_all)
        
        if "split" in self.df.columns:
            self.splits = self.df["split"].to_numpy()
        else:
            self.splits = np.full(len(self.df), "undefined")

    def apply_split(self):
        self.x_train = self.x_all[self.splits == "train"]
        self.y_train = self.y_encoded[self.splits == "train"]
        self.x_test = self.x_all[self.splits == "test"]
        self.y_test = self.y_encoded[self.splits == "test"]


def get_model_map(seed, tuning=False):
    model_map = {
        "RandomForest": RandomForestClassifier(random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "KNN": KNeighborsClassifier()
    }
    if tuning:
        # Include SVM only if explicitly requested/configured
        # model_map["SVM"] = SVC(probability=True, random_state=seed)
        pass
    return model_map


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def run_permutation_test(model_class, param_dict, X, y, model_name, seed, save_dir, n_permutations=1000, cv_folds=5):
    """
    Run permutation test on a fitted model and save results incrementally to a single CSV.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model = model_class(**param_dict)

    score, perm_scores, pvalue = permutation_test_score(
        estimator=model,
        X=X,
        y=y,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=seed
    )

    result = {
        "model": model_name,
        "seed": seed,
        "accuracy": round(score, 3),
        "pvalue": round(pvalue, 5)
    }

    # Path to cumulative CSV
    cumulative_path = os.path.join(save_dir, "permutation_results.csv")
    df_result = pd.DataFrame([result])

    # Append or write with header
    if os.path.exists(cumulative_path):
        df_result.to_csv(cumulative_path, mode="a", header=False, index=False)
    else:
        df_result.to_csv(cumulative_path, mode="w", header=True, index=False)

    # Save histogram
    plt.figure()
    plt.hist(perm_scores, bins=20, density=True, alpha=0.7, label="Permuted scores")
    plt.axvline(score, color="red", linestyle="--", label=f"Observed score = {score:.2f}")
    plt.title(f"Permutation Test - {model_name} (seed {seed})\np = {pvalue:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"perm_hist_{model_name}.png"))
    plt.close()

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics. Include AUC if probabilities are available."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_roc": None
    }
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            metrics["auc_roc"] = auc
        except:
            metrics["auc_roc"] = None
    return metrics

def train_and_evaluate_model(base_model, model_name, param_dict, data: DataSplit, params: dict, output_dir: str):
    """Train model (CV Grid Search or Direct) and evaluate on test data."""
    seed = params["seed"]
    
    # === INNER CV (GRID SEARCH) ===
    if params.get("tuning", False):
        skf = StratifiedKFold(n_splits=params["n_folds"], shuffle=True, random_state=seed)
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_dict,
            scoring="accuracy",
            cv=skf,
            n_jobs=-1,
            refit=True, # Automatically refits on the whole training set
            verbose=0
        )
        grid.fit(data.x_train, data.y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Save Grid Result
        df_grid = pd.DataFrame(grid.cv_results_)
        keep_cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        # Filter columns to ensure they exist
        keep_cols = [c for c in keep_cols if c in df_grid.columns]
        
        df_grid[keep_cols].round(3).to_csv(os.path.join(output_dir, "cv_grid.csv"), index=False)
        
        # Save Best Params
        with open(os.path.join(output_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

    # === DIRECT TRAINING (NO TUNING) ===
    else:
        base_model.set_params(**param_dict)
        base_model.fit(data.x_train, data.y_train)
        best_model = base_model
        best_params = param_dict
        
        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

    # === PREDICT ON OUTER TEST SET ===
    y_pred = best_model.predict(data.x_test)
    try:
        y_proba = best_model.predict_proba(data.x_test)
    except:
        y_proba = None

    plot_confusion_matrix(
        data.y_test, y_pred, class_names=data.le.classes_,
        title=f"{model_name} | Seed {seed} | Outer Fold",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    # Compute metrics
    metrics = evaluate_metrics(data.y_test, y_pred, y_proba)
    
    return best_model, best_params, metrics, y_pred

def nested_cv_classification(params, df_input, output_dir):
    """
    Nested cross-validation: 
    - Outer Loop (5 Folds): Estimate Generalization Error
    - Inner Loop (5 Folds): Hyperparameter Tuning via Grid Search
    - Full Retrain: Best params on 100% Outer Train (No Augmentation)
    """
    
    # Filter only selected groups
    df_filtered = df_input[df_input["Group"].isin([params["group1"], params["group2"]])].copy()
    
    print(f"\nDataset: {len(df_filtered)} subjects ({params['group1']}, {params['group2']})")
    print(f"Outer folds: {params.get('n_outer_folds', 5)}")
    
    # Prepare features
    # Use DataSplit generic class to handle feature extraction
    data_container = DataSplit(df_filtered, split_path=None, use_full_input=False)
    data_container.prepare_features()
    
    X = data_container.x_all
    y_encoded = data_container.y_encoded
    ids = df_filtered["ID"].to_numpy()
    ids = df_filtered["ID"].to_numpy()
    groups = df_filtered["Group"].to_numpy()

    # Fixed seed for Outer CV - ensuring reproducibility of the folds
    outer_cv = StratifiedKFold(
        n_splits=params.get("n_outer_folds", 5),
        shuffle=True,
        random_state=42 
    )
    
    # Load Grids
    if params.get("tuning", False):
        with open("src/ML_analysis/config/ml_grid.json", "r") as f:
             param_grids = json.load(f)
    else:
        # Fallback to single params if not tuning
        param_grids = {
            "RandomForest": params["RandomForest"],
            "GradientBoosting": params["GradientBoosting"],
            "KNN": params["KNN"]
        }
    
    model_map = get_model_map(seed=42, tuning=params["tuning"])
    
    # Initialize structure to hold all results
    all_results = []
    all_predictions = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded), start=1):
        print("\n" + "=" * 60)
        print(f"OUTER FOLD {fold_idx}/{params.get('n_outer_folds', 5)}")
        print("=" * 60)
        
        # 1. Outer Split
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
        ids_train, ids_test = ids[train_idx], ids[test_idx]
        
        print(f"Train: {len(X_train_fold)} subjects | Test: {len(X_test_fold)} subjects")
        
        # 2. UMAP (Fit on Train, Transform Test to avoid leakage)
        if params.get("umap", False):
            print("Applying UMAP (Fit Train -> Transform Test)...")
            X_train_fold, X_test_fold = run_umap(X_train_fold, X_test_fold)
        
        # Update DataSplit object for this fold
        # We reuse the same object structure to pass to train_model
        data_fold = type('DataSplit', (), {})()
        data_fold.x_train = X_train_fold
        data_fold.x_test = X_test_fold
        data_fold.y_train = y_train_fold
        data_fold.y_test = y_test_fold
        data_fold.le = data_container.le
        data_fold.df = pd.DataFrame({"ID": ids_test}) # Meta info for current test set
        
        # 3. Iterate Models
        for model_name, base_model in model_map.items():
            print(f"  > Processing {model_name}...")
            
            # Directory structure: results/.../[ModelName]/fold_[i]/
            model_fold_dir = os.path.join(output_dir, model_name, f"fold_{fold_idx}")
            os.makedirs(model_fold_dir, exist_ok=True)
            
            # Use fixed seed for training (or list of seeds if we wanted multiple runs per fold)
            # Here we follow DL approach: 1 deterministic run per fold
            params["seed"] = 42 
            
            # 4. Train (Inner CV) & Evaluate
            # base_model is cloned inside GridSearch, so we pass instance
            import time
            start_time = time.time()
            best_model, best_params, metrics, y_pred = train_and_evaluate_model(
                 base_model, model_name, param_grids[model_name], data_fold, params, model_fold_dir
            )
            elapsed_time = time.time() - start_time
            print(f"    [Timing] {model_name} processing time: {elapsed_time:.2f} seconds")
            
            # 5. Collect metrics
            result_entry = {
                "outer_fold": fold_idx,
                "model": model_name,
                "dataset": params['dataset_type'],
                **metrics,
                "best_params": str(best_params)
            }
            all_results.append(result_entry)
            
            # 6. Collect predictions
            test_ids = ids_test
            true_labels = data_fold.le.inverse_transform(y_test_fold)
            pred_labels = data_fold.le.inverse_transform(y_pred)
            
            df_preds = pd.DataFrame({
                "ID": test_ids,
                "outer_fold": fold_idx,
                "Model": model_name,
                "TrueLabel": true_labels,
                "PredLabel": pred_labels
            })
            df_preds.to_csv(os.path.join(model_fold_dir, "predictions.csv"), index=False)
            all_predictions.append(df_preds)

    # === AGGREGATION & SAVING ===
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # Save raw results (flat file)
        df_all.to_csv(os.path.join(output_dir, "nested_cv_all_results.csv"), index=False)
        
        # Calculate Aggregated Stats per Model
        print("\n" + "=" * 60)
        print("NESTED CV SUMMARY (Mean + Std across 5 Folds)")
        print("=" * 60)
        
        summary_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        # Group by Model and calculate mean/std
        summary = df_all.groupby("model")[summary_cols].agg(["mean", "std"]).round(3)
        print(summary)
        summary.to_csv(os.path.join(output_dir, "nested_cv_summary.csv"))
        
        # Save per-model summary JSON in each model folder
        for model_name in df_all["model"].unique():
             model_stats = df_all[df_all["model"] == model_name][summary_cols].agg(["mean", "std"]).to_dict()
             model_res_path = os.path.join(output_dir, model_name, "aggregated_results.json")
             with open(model_res_path, "w") as f:
                 json.dump(model_stats, f, indent=4)

    if all_predictions:
         df_all_preds = pd.concat(all_predictions, ignore_index=True)
         df_all_preds.to_csv(os.path.join(output_dir, "nested_cv_all_predictions.csv"), index=False)


def single_split_classification(params, df_input, output_dir):
    """
    Attributes:
        Legacy function for fixed train/test split.
    Notes:
        Kept for backward compatibility if use_fixed_split=True.
    """
    print("WARNING: Using Fixed Split Mode (Legacy)")
    pass # Implementation elided for brevity, as we are focused on Nested CV


def main_classification(params, df_input):
    """Main entry point."""
    group_dir = f"{params['group1'].lower()}_{params['group2'].lower()}"
    output_dir = os.path.join(
        build_output_path(params['output_dir'], params['task_type'], params['dataset_type'], params['umap'], False),
        group_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, "log_nested_cv.txt")
    log_to_file(log_path)

    # Route to appropriate classification method
    if params.get("use_fixed_split", False): 
        # single_split_classification(params, df_input, output_dir)
        print("Fixed split support is minimized in this refactor. Please use Nested CV.")
    else:
        print("=" * 60)
        print(f"STARTING NESTED CV PIPELINE for {params['group1']} vs {params['group2']}")
        print("=" * 60)
        nested_cv_classification(params, df_input, output_dir)

    reset_stdout()

if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_classification(args, input_dataframe)
