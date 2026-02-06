
import os
import csv
import glob
import ast
import pandas as pd

# Define paths
base_ml_path = "/data/users/etosato/ANM_Verona/results/ML"
paths = {
    "UMAP-based": os.path.join(base_ml_path, "voxel", "umap_classification"),
    "Atlas-based": os.path.join(base_ml_path, "networks", "classification")
}

# Metrics to report (User just wants Accuracy for this table, plus Params)
models_map = {
    "GB": "GradientBoosting",
    "KNN": "KNN",
    "RF": "RandomForest"
}
comparisons = ["AD_PSP", "AD_CBS", "PSP_CBS"] # Check dir names

# Ensure output directory exists
os.makedirs(os.path.join(base_ml_path), exist_ok=True)
csv_output_path = os.path.join(base_ml_path, "best_params_table.csv")

with open(csv_output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for method_name, method_path in paths.items():
        if not os.path.exists(method_path):
            continue
            
        # Write Section Header
        writer.writerow([method_name])
        writer.writerow(["Comparison", "Model", "Best Accuracy (Fold)", "Optimal Hyperparameters"])

        for comp in comparisons:
            comp_dir = os.path.join(method_path, comp)
            if not os.path.exists(comp_dir):
                comp_hyphen = comp.replace("_", "-")
                comp_dir = os.path.join(method_path, comp_hyphen)
                if not os.path.exists(comp_dir):
                     continue
                comp_label = comp_hyphen
            else:
                comp_label = comp.replace("_", "–") # Use en-dash
                
            for model_abbr, model_name in models_map.items():
                model_dir = os.path.join(comp_dir, model_name)
                result_file = os.path.join(model_dir, "nested_cv_results.csv")
                
                acc = ""
                cleaned_params = ""

                if os.path.exists(result_file):
                    try:
                        df = pd.read_csv(result_file)
                        best_fold = df.loc[df['accuracy'].idxmax()]
                        acc = f"{best_fold['accuracy']:.3f}"
                        params_str = best_fold['best_params']
                        try:
                            params_dict = ast.literal_eval(params_str)
                            cleaned_params = ", ".join([f"{k}={v}" for k,v in params_dict.items()])
                        except:
                            cleaned_params = params_str
                    except Exception as e:
                        cleaned_params = f"Error: {e}"
                
                writer.writerow([comp_label, model_abbr, acc, cleaned_params])
        
        writer.writerow([]) # Empty row between sections

print(f"CSV created at: {csv_output_path}")
