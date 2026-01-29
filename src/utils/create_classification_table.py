import os
import pandas as pd
import csv

def main():
    base_dir = "/data/users/etosato/ANM_Verona/results/ML"
    output_file = os.path.join(base_dir, "classification_table.csv")
    
    datasets = [
        {
            "name": "VOXEL",
            "results_path": os.path.join(base_dir, "voxel/umap_classification/total_aggregated_results.csv"),
            "perm_base": os.path.join(base_dir, "voxel/umap_classification")
        },
        {
            "name": "NETWORK",
            "results_path": os.path.join(base_dir, "networks/classification/total_aggregated_results.csv"),
            "perm_base": os.path.join(base_dir, "networks/classification")
        }
    ]
    
    # Comparisons order
    comparisons = ["AD_CBS", "AD_PSP", "PSP_CBS"]
    
    # Models order
    models = ["RandomForest", "GradientBoosting", "KNN"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for ds in datasets:
            ds_name = ds['name']
            writer.writerow([ds_name])
            
            # Load main results
            if os.path.exists(ds['results_path']):
                # Read CSV expecting headers. Row 0 is standard header. Row 1 is mean/std (data row 0).
                df_res = pd.read_csv(ds['results_path'])
            else:
                writer.writerow(["(Results file not found)"])
                writer.writerow([])
                continue
            
            # Header
            writer.writerow(["Comparison", "Model", "Accuracy", "F1", "AUC", "P-Value"])
            
            for comp in comparisons:
                # Format comparison string for matching (e.g. AD_CBS -> AD vs CBS)
                target_comp = comp.replace('_', ' vs ')
                
                # Load permutation stats for this comparison
                perm_file = os.path.join(ds['perm_base'], comp, "permutation_stats1000.csv")
                
                df_perm = None
                if os.path.exists(perm_file):
                    df_perm = pd.read_csv(perm_file)
                
                first_row = True
                for model in models:
                    row = []
                    
                    # Comparison Label (only on first row of group)
                    if first_row:
                        row.append(comp)
                        first_row = False
                    else:
                        row.append("")
                        
                    row.append(model)
                    
                    # Get Metrics from df_res
                    # Layout in total_aggregated_results.csv:
                    # Col 0: model
                    # Col 1: comparison
                    # Col 2: accuracy (mean)
                    # Col 3: accuracy (std)
                    # Col 4: precision (mean)
                    # Col 5: precision (std)
                    # Col 6: recall (mean)
                    # Col 7: recall (std)
                    # Col 8: f1 (mean)
                    # Col 9: f1 (std)
                    # Col 10: auc_roc (mean)
                    # Col 11: auc_roc (std)
                    
                    matched_row = df_res[
                        (df_res['comparison'] == target_comp) & 
                        (df_res.iloc[:, 0] == model)
                    ]
                    
                    if not matched_row.empty:
                        try:
                            # Accuracy
                            acc_mean = matched_row.iloc[0, 2]
                            acc_std = matched_row.iloc[0, 3]
                            # F1
                            f1_mean = matched_row.iloc[0, 8]
                            f1_std = matched_row.iloc[0, 9]
                            # AUC
                            auc_mean = matched_row.iloc[0, 10]
                            auc_std = matched_row.iloc[0, 11]
                            
                            row.append(f"{acc_mean} ({acc_std})")
                            row.append(f"{f1_mean} ({f1_std})")
                            row.append(f"{auc_mean} ({auc_std})")
                        except IndexError:
                             row.extend(["Error", "Error", "Error"])
                    else:
                        # Fallback if not found
                        row.extend(["N/A", "N/A", "N/A"])
                        
                    # Get P-Value from df_perm
                    if df_perm is not None:
                        # Find row where model == model
                        perm_row = df_perm[df_perm['model'] == model]
                        if not perm_row.empty:
                            p_val = perm_row.iloc[0]['p_value']
                            try:
                                p_val = f"{float(p_val):.4f}"
                            except:
                                pass
                            row.append(p_val)
                        else:
                            row.append("N/A")
                    else:
                        row.append("Running..." if comp == "PSP_CBS" and ds_name == "VOXEL" else "N/A")
                    
                    writer.writerow(row)
                    
                writer.writerow([]) # Separator between comparisons
                
            writer.writerow([])
            writer.writerow([])

    print(f"Classification table written to {output_file}")

if __name__ == "__main__":
    main()
