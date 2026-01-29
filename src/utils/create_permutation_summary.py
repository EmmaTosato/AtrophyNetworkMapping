import os
import pandas as pd
import csv

def main():
    base_dir = "/data/users/etosato/ANM_Verona/results/ML"
    output_file = os.path.join(base_dir, "total_permutations_results.csv")
    
    # Define roots to search (Voxel first as requested)
    roots = [
        ("Voxel", os.path.join(base_dir, "voxel/umap_classification")),
        ("Network", os.path.join(base_dir, "networks/classification"))
    ]
    
    aggregated_dfs = []
    
    for dataset_name, root_path in roots:
        if not os.path.exists(root_path):
            continue
            
        # Walk to find permutation_stats1000.csv
        for dirpath, dirs, files in os.walk(root_path):
            for f in files:
                if f == "permutation_stats1000.csv":
                    full_path = os.path.join(dirpath, f)
                    try:
                        df = pd.read_csv(full_path, dtype=str) # Read as string to preserve formatting
                        if not df.empty:
                            df['Dataset'] = dataset_name
                            
                            # Clean up comparison name from directory or file content?
                            # Usually directory name is 'AD_CBS'
                            # File has 'groups' col?
                            if 'groups' in df.columns:
                                df['Comparison'] = df['groups'].str.replace('vs', ' vs ') # ensure spacing
                                # Wait, file content in Voxel PSP_CBS was 'PSPvsCBS' -> 'PSP vs CBS'
                            else:
                                # Fallback to directory name
                                comp_name = os.path.basename(dirpath)
                                df['Comparison'] = comp_name.replace('_', ' vs ')
                                
                            aggregated_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

    if not aggregated_dfs:
        print("No permutation stats found.")
        return

    full_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    # Columns to keep and rename
    # Original: dataset,groups,model,n_perms,true_score,p_value,perm_scores_mean,perm_scores_std
    # Desired: Dataset, Comparison, Model, Number of Permutations, True Score, P-Value, Permutation Score (Mean ± Std)
    
    # Clean Model Names
    # GradientBoosting -> GB, RandomForest -> RF, KNN -> KNN
    def clean_model(m):
        if m == "GradientBoosting": return "GB"
        if m == "RandomForest": return "RF"
        return m
    
    full_df['Model'] = full_df['model'].apply(clean_model)
    
    # Merge Mean/Std
    def merge_mean_std(row):
        m = row.get('perm_scores_mean', 'N/A')
        s = row.get('perm_scores_std', 'N/A')
        return f"{m} ± {s}"
    
    full_df['Permutation Score (Mean ± Std)'] = full_df.apply(merge_mean_std, axis=1)
    
    # Rename simple cols
    full_df.rename(columns={
        'n_perms': 'Number of Permutations',
        'true_score': 'True Score',
        'p_value': 'P-Value'
    }, inplace=True)
    
    # Select final columns
    final_cols = ['Dataset', 'Comparison', 'Model', 'Number of Permutations', 'True Score', 'P-Value', 'Permutation Score (Mean ± Std)']
    
    # Ensure Comparison format is uniform
    # Some might be 'AD_CBS' (Network) vs 'ADvsCBS' (Voxel file content).
    # Let's normalize to 'AD vs CBS'
    def norm_comp(c):
        c = str(c).replace('_', ' vs ')
        if 'vs' in c and ' vs ' not in c:
            c = c.replace('vs', ' vs ')
        return c
    full_df['Comparison'] = full_df['Comparison'].apply(norm_comp)
    
    # Sort
    # Enforce Voxel before Network
    full_df['Dataset'] = pd.Categorical(full_df['Dataset'], categories=['Voxel', 'Network'], ordered=True)
    
    full_df.sort_values(by=['Dataset', 'Comparison', 'Model'], inplace=True)
    
    # Write to CSV with grouping logic
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(final_cols)
        
        last_dataset = None
        last_comp = None
        
        for _, row in full_df.iterrows():
            current_dataset = row['Dataset']
            current_comp = row['Comparison']
            
            display_dataset = current_dataset
            display_comp = current_comp
            
            # Hide repeated Dataset
            if current_dataset == last_dataset:
                display_dataset = ""
                # Hide repeated Comparison (only if Dataset is also same)
                if current_comp == last_comp:
                    display_comp = ""
                else:
                    last_comp = current_comp
            else:
                last_dataset = current_dataset
                last_comp = current_comp
                # Separator line if dataset changes? (Optional)
                
            out_row = [display_dataset, display_comp]
            # Add rest of columns
            for col in final_cols[2:]:
                out_row.append(row[col])
                
            writer.writerow(out_row)

    print(f"Permutations table written to {output_file}")

if __name__ == "__main__":
    main()
