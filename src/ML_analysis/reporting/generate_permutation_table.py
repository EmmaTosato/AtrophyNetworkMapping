import pandas as pd
import os
import json
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_perm_stats(ds_path, resolved_comp_dir, model_name):
    path = os.path.join(ds_path, resolved_comp_dir, model_name, "permutation_stats.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path).iloc[0]
        except:
            pass
    return None

def generate_perm_table(config_path="src/ML_analysis/config/report_config.json", output_file="results/ML/summary_permutation.csv"):
    config = load_config(config_path)
    datasets = config["datasets"]
    comparisons = config["comparisons"]
    models = config["models"]
    model_map = config["model_display_names"]
    
    rows = []
    
    for ds_info in datasets:
        ds_name = ds_info["name"]
        ds_path = ds_info["path"]
        
        # Section Header Row
        rows.append({
            "Comparison": f"--- {ds_name} ---",
            "Model": "",
            "True Accuracy": "",
            "P-Value": "",
            "Mean Null Acc": "",
            "Null Std": ""
        })
        
        for comp in comparisons:
            comp_lower = comp.lower()
            possible_namings = [comp_lower, f"{comp_lower.split('_')[1]}_{comp_lower.split('_')[0]}"]
            
            resolved_dir = None
            for n in possible_namings:
                # Check existance of ANY model folder to confirm directory
                # or check if 'csv_results' exists
                if os.path.exists(os.path.join(ds_path, n)):
                    resolved_dir = n
                    break
            
            if not resolved_dir:
                continue
                
            first_row = True
            for model in models:
                stats = load_perm_stats(ds_path, resolved_dir, model)
                
                m_disp = model_map.get(model, model)
                comp_disp = comp.replace('_', ' vs ') if first_row else ""
                
                if stats is None:
                    row = {
                        "Comparison": comp_disp,
                        "Model": m_disp,
                        "True Accuracy": "Pending",
                        "P-Value": "",
                        "Mean Null Acc": "",
                        "Null Std": ""
                    }
                else:
                    # Format values
                    true_sc = f"{stats.get('true_score', 0):.4f}"
                    pval = stats.get('p_value', 1.0)
                    mean_null = f"{stats.get('perm_scores_mean', 0):.4f}" # Check actual col name
                    # In previous script we saved: perm_scores_mean, perm_scores_std. 
                    # Let's check consistency. The raw file consolidation showed 'perm_scores_mean'.
                    null_std = f"{stats.get('perm_scores_std', 0):.4f}" if 'perm_scores_std' in stats else ""
                    
                    pval_str = f"{pval:.4f}"
                    if pval < 0.001: pval_str = "< 0.001"
                    
                    row = {
                        "Comparison": comp_disp,
                        "Model": m_disp,
                        "True Accuracy": true_sc,
                        "P-Value": pval_str,
                        "Mean Null Acc": mean_null,
                        "Null Std": null_std
                    }
                
                rows.append(row)
                first_row = False

    df = pd.DataFrame(rows)
    # Reorder cols
    df = df[["Comparison", "Model", "True Accuracy", "P-Value", "Mean Null Acc", "Null Std"]]
    
    df.to_csv(output_file, index=False)
    print(f"Saved formatted permutation table to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ML/summary_permutation.csv")
    args = parser.parse_args()
    
    generate_perm_table(output_file=args.out)
