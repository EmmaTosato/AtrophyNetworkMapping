
import os
import pandas as pd
import json

base_dir = "/data/users/etosato/ANM_Verona/results/DL"
config_file = os.path.join(base_dir, "global_config.json")
output_file = os.path.join(base_dir, "best_params_summary.csv")

# Load Global Config
global_params = {}
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        cfg = json.load(f)
        grids = cfg.get("grids", {})
        # Flatten lists since they are single-valued
        for model, p_grid in grids.items():
            fixed = {}
            for k, v in p_grid.items():
                if isinstance(v, list) and len(v) > 0:
                    fixed[k] = v[0]
                else:
                    fixed[k] = v
            global_params[model] = fixed

summary_list = []
comparisons = ["AD_PSP", "AD_CBS", "PSP_CBS"]
target_models = ["alexnet", "resnet", "vgg16"]

for contrast in comparisons:
    contrast_path = os.path.join(base_dir, contrast)
    if not os.path.exists(contrast_path):
        continue
        
    for model in target_models:
        model_path = os.path.join(contrast_path, model)
        csv_path = os.path.join(model_path, "nested_cv_results.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # lowercase cols
                df.columns = [c.lower() for c in df.columns]
                
                if 'accuracy' in df.columns:
                    best_row = df.loc[df['accuracy'].idxmax()]
                    best_fold = best_row.get('fold', 'N/A')
                    accuracy = best_row['accuracy']
                    
                    # Get fixed params
                    # Handle renaming (resnet vs resnet18 etc)
                    # Config uses "resnet", folder uses "resnet"
                    # Config "vgg16", folder "vgg16"
                    cur_params = global_params.get(model, {})
                    
                    summary_list.append({
                        "Comparison": contrast,
                        "Model": model,
                        "Best_Fold": int(best_fold) if best_fold != 'N/A' else 'N/A',
                        "Accuracy": round(accuracy, 4),
                        "Precision": round(best_row.get('precision', 0), 4),
                        "Recall": round(best_row.get('recall', 0), 4),
                        "F1_Score": round(best_row.get('f1', 0), 4),
                        "AUC": round(best_row.get('auc', 0), 4),
                        "Learning_Rate": cur_params.get("lr"),
                        "Batch_Size": cur_params.get("batch_size"),
                        "Weight_Decay": cur_params.get("weight_decay"),
                        "Epochs": cur_params.get("epochs"),
                        "Optimizer": cur_params.get("optimizer")
                    })
            except Exception as e:
                print(f"Error {csv_path}: {e}")

df_summary = pd.DataFrame(summary_list)
# Reorder columns for clarity
cols = ["Comparison", "Model", "Best_Fold", "Accuracy", "Precision", "Recall", "F1_Score", "AUC", 
        "Learning_Rate", "Batch_Size", "Weight_Decay", "Epochs", "Optimizer"]
df_summary = df_summary[cols]

df_summary.to_csv(output_file, index=False)
print(f"Summary saved to {output_file}")
print(df_summary.to_string())
