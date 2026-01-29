import os
import pandas as pd
import numpy as np
import csv

def format_series_value(x):
    # Try float conversion
    try:
        f = float(x)
        # Check if it's an integer (like y_pred often is 0 or 1, or fold index)
        # If it's effectively an integer (1.0), maybe keep as int?
        # User asked for "3 cifre decimali" for results.
        # But prediction labels (0, 1) shouldn't be 0.000, 1.000 probably.
        # However, `y_prob` should be.
        # Results metrics (accuracy) should be.
        
        # Let's be aggressive with formatting as requested, but maybe spare obvious integers/indices?
        # Use simple heuristic: if col name contains 'pred' or 'true' or 'fold' or 'label', might be int.
        # But user cited `nested_cv_results.csv`.
        
        # Let's just format ALL floats.
        return f"{f:.3f}"
    except (ValueError, TypeError):
        return x

def format_csv_file(filepath):
    # Skip aggregated file during this pass, or format it too?
    # Format it too is safer.
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return

    modified = False
    
    # Identify numeric columns
    # We want to format them to strings.
    
    # Iterate columns
    for col in df.columns:
        # Check if column is numeric or object (could be stringified floats)
        # We try to convert to numeric first to handle mixed types
        # But preserve original if it fails.
        
        # heuristic: special columns to SKIP formatting?
        # 'fold_idx', 'y_true', 'y_pred' (if distinct classes)
        # But 'y_pred' for classification is 0/1. 0.000 is ugly.
        # User complained about `nested_cv_results.csv` which has metrics.
        
        is_integer_like = False
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # Check if all non-NaN values are integers
            non_nan = numeric_series.dropna()
            if len(non_nan) > 0 and (non_nan % 1 == 0).all():
                is_integer_like = True
        except:
            pass

        if 'iter' in col.lower() or 'fold' in col.lower() or 'n_perms' in col.lower():
            is_integer_like = True
            
        if not is_integer_like:
            # Format
            # We explicitly apply the formatter to the Series
            # We convert to numeric first to ensure we catch strings "0.123"
            numeric_s = pd.to_numeric(df[col], errors='coerce')
            
            # Apply formatting only to non-NaN entries in this numeric series
            # If original was NaN or non-numeric string, it remains.
            
            # We construct a new series
            new_col = numeric_s.apply(lambda x: f"{x:.3f}" if pd.notnull(x) else np.nan)
            
            # Fill NaN in new_col with original values (for non-numeric strings that became NaNs)
            # This is tricky. simpler: apply map function to original column?
            
            df[col] = df[col].apply(format_series_value)
            modified = True
        else:
            # Ensure integers look like integers (no 1.0)
            if df[col].dtype == 'float':
                 df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) and x==x else x)
                 modified = True

    if modified:
        df.to_csv(filepath, index=False)
        print(f"  Formatted: {filepath}")

def aggregate_dataset_results(base_dir):
    print(f"Aggregating in {base_dir}...")
    
    aggregated_rows = []
    
    try:
        items = os.listdir(base_dir)
    except FileNotFoundError:
        return

    comparisons = []
    for item in items:
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and ("_" in item or " vs " in item): 
             comparisons.append(item)
    
    comparisons.sort()
    
    # Column mapping for aggregation
    col_map = {
        'accuracy': ('accuracy', 'mean'),
        'accuracy.1': ('accuracy', 'std'),
        'precision': ('precision', 'mean'),
        'precision.1': ('precision', 'std'),
        'recall': ('recall', 'mean'),
        'recall.1': ('recall', 'std'),
        'f1': ('f1', 'mean'),
        'f1.1': ('f1', 'std'),
        'auc_roc': ('auc_roc', 'mean'),
        'auc_roc.1': ('auc_roc', 'std')
    }
    metric_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']

    for comp in comparisons:
        comp_dir = os.path.join(base_dir, comp)
        summary_file = os.path.join(comp_dir, "summary_results.csv")
        
        if os.path.exists(summary_file):
            # Read as string to preserve formatting (e.g. 0.110)
            df = pd.read_csv(summary_file, dtype=str)
            
            # Identify model column (first column)
            model_col = df.columns[0]
            
            def is_valid_row(val):
                s = str(val)
                if s == 'nan' or s == 'None': return False
                if 'mean' in s: return False
                if 'model' in s: return False
                if s.strip() == '': return False
                return True
            
            df_clean = df[df[model_col].apply(is_valid_row)].copy()
            df_clean.rename(columns={model_col: "model"}, inplace=True)
            df_clean['comparison'] = comp # Keep original dirname or format?
            # User wants " vs ". If dirname is AD_CBS, formatting happens later or here.
            
            aggregated_rows.append(df_clean)

    if not aggregated_rows:
        return

    full_df = pd.concat(aggregated_rows, ignore_index=True)
    
    output_path = os.path.join(base_dir, "total_aggregated_results.csv")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        h1 = ['comparison', 'model']
        # h2 not needed anymore for single header row
        
        # Define the metrics we want to output
        final_metrics = metric_order # ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        h1.extend(final_metrics)
        
        writer.writerow(h1)
        # writer.writerow(h2) # Removing second header row
        
        # Sort and Write
        full_df.sort_values(by=['comparison', 'model'], inplace=True)
        
        last_comp = None
        for _, row in full_df.iterrows():
            current_comp = row['comparison'].replace('_', ' vs ')
            display_comp = current_comp
            
            if current_comp == last_comp:
                display_comp = ""
            else:
                last_comp = current_comp
            
            model_name = row['model']
            if model_name == "GradientBoosting": model_name = "GB"
            if model_name == "RandomForest": model_name = "RF"
            
            out_row = [display_comp, model_name]
            
            # Construct merged metric strings
            for m in final_metrics:
                m_mean_col = None
                m_std_col = None
                
                # Find the column names in df for this metric's mean/std
                for c_name, (metric_name, metric_type) in col_map.items():
                    if metric_name == m and metric_type == 'mean' and c_name in full_df.columns: m_mean_col = c_name
                    if metric_name == m and metric_type == 'std' and c_name in full_df.columns: m_std_col = c_name
                
                val_str = "N/A"
                if m_mean_col and m_std_col:
                    mean_val = row[m_mean_col]
                    std_val = row[m_std_col]
                    val_str = f"{mean_val} ± {std_val}"
                elif m_mean_col:
                    val_str = f"{row[m_mean_col]}"
                
                out_row.append(val_str)
            
            writer.writerow(out_row)
            
    print(f"  Aggregated: {output_path}")

def process_tree(root_dir):
    print(f"Processing root: {root_dir}")
    
    # 1. Recursive format of ALL CSVs
    for dirpath, dirs, files in os.walk(root_dir):
        # Exclude 'fold' directories
        if 'fold' in dirpath:
            continue
        
        dirs[:] = [d for d in dirs if 'fold' not in d] # prevent descending into folds
        
        for f in files:
            if f.endswith(".csv"):
                # Special check: do not format 'total_aggregated_results.csv' yet?
                # Actually, treating it as a normal CSV might break its multi-header structure 
                # if we read/write it simply.
                # It's better to regenerate it at the end.
                if f == "total_aggregated_results.csv":
                    continue
                
                # Also skip 'classification_table.csv' if it exists there
                if f == "classification_table.csv":
                    continue
                    
                path = os.path.join(dirpath, f)
                format_csv_file(path)

    # 2. Aggregate
    aggregate_dataset_results(root_dir)

def combine_global_classification_results():
    base_dir = "/data/users/etosato/ANM_Verona/results/ML"
    output_file = os.path.join(base_dir, "total_classification_results.csv")
    
    # Define sources (Voxel first)
    sources = [
        ("Voxel", os.path.join(base_dir, "voxel/umap_classification/total_aggregated_results.csv")),
        ("Network", os.path.join(base_dir, "networks/classification/total_aggregated_results.csv"))
    ]
    
    combined_dfs = []
    
    for ds_name, path in sources:
        if os.path.exists(path):
            try:
                # Read as string to preserve formatting, empty strings as empty strings
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                # Add Dataset column at the beginning
                df.insert(0, 'dataset', ds_name)
                combined_dfs.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                
    if not combined_dfs:
        return

    full_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Write to CSV
    # We want grouping logic similar to other tables
    # Group by Dataset, then Comparison
    
    # Columns: dataset, comparison, model, ...
    cols = list(full_df.columns)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        
        last_dataset = None
        last_comp = None
        
        for _, row in full_df.iterrows():
            current_dataset = row['dataset']
            current_comp = row['comparison']
            
            display_dataset = current_dataset
            display_comp = current_comp
            
            if current_dataset == last_dataset:
                display_dataset = ""
                # Check comparison
                if current_comp == last_comp:
                    display_comp = ""
                else:
                    last_comp = current_comp
            else:
                last_dataset = current_dataset
                last_comp = current_comp # Reset comparison tracker for new dataset
            
            out_row = [display_dataset, display_comp]
            # Add rest
            for col in cols[2:]:
                out_row.append(row[col])
                
            writer.writerow(out_row)
            
    print(f"Global Combined Classification table written to {output_file}")

def main():
    root = "/data/users/etosato/ANM_Verona/results/ML"
    path_net = os.path.join(root, "networks/classification")
    path_vox = os.path.join(root, "voxel/umap_classification")
    
    if os.path.exists(path_net): process_tree(path_net)
    if os.path.exists(path_vox): process_tree(path_vox)
    
    # Combine everything globally
    combine_global_classification_results()

if __name__ == "__main__":
    main()
