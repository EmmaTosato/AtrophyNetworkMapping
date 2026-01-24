import pandas as pd
import os
import glob
import argparse

def consolidate_nested_cv(base_dir, output_file):
    print(f"Searching for nested_cv_summary.csv in {base_dir}...")
    
    # Recursive search
    files = glob.glob(os.path.join(base_dir, "**", "nested_cv_summary.csv"), recursive=True)
    
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Merging...")
    
    dfs = []
    for f in files:
        try:
            # Read with multi-level header
            df = pd.read_csv(f, header=[0, 1], index_col=0)
            
            # Reset index to make 'model' a column
            df = df.reset_index()
            # If index name was missing, rename first col to 'model'
            if df.columns[0][0] == 'index':
                 df.rename(columns={df.columns[0]: ('model', '')}, inplace=True)
            
            # Flatten columns
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    metric, stat = col
                    if stat:
                        if "Unnamed" in metric: metric = "model" # Fix for index column if weird
                        new_cols.append(f"{metric}_{stat}")
                    else:
                        new_cols.append(metric) # e.g. model
                else:
                    new_cols.append(col)
            
            df.columns = new_cols
            
            # Identify Comparison from folder path
            # Structure: .../ad_cbs/csv_results/nested_cv_summary.csv
            # Parent of parent is usually comp name
            parent = os.path.dirname(f)
            grandparent = os.path.dirname(parent)
            comp_name = os.path.basename(grandparent)
            
            # If path doesn't match standard structure, try to be robust
            if "_" not in comp_name and len(comp_name) < 3:
                # Maybe file was in root .../ad_cbs/nested_cv_summary.csv
                comp_name = os.path.basename(parent)
            
            df["comparison"] = comp_name
            
            # Clean up column names (ensure 'model' is consistent)
            # rename first col if it looks like the model column
            if "model" not in df.columns and "index" not in df.columns:
                 # Usually the first column is the model
                 df.rename(columns={df.columns[0]: "model"}, inplace=True)
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        
        # Reorder columns to put comparison and model first
        cols = list(master_df.columns)
        first_cols = ["comparison", "model"]
        other_cols = [c for c in cols if c not in first_cols]
        master_df = master_df[first_cols + other_cols]

        # Round numeric
        numeric_cols = master_df.select_dtypes(include=['float', 'float64']).columns
        master_df[numeric_cols] = master_df[numeric_cols].round(4)
        
        # Sort by comparison to ensure grouping
        master_df.sort_values(by="comparison", inplace=True)

        # Insert spacing rows for better readability
        formatted_rows = []
        prev_comp = None
        for _, row in master_df.iterrows():
            curr_comp = row['comparison']
            # Insert spacer if group changes (and not first row)
            if prev_comp is not None and curr_comp != prev_comp:
                # Create empty row dict
                spacer = {c: "" for c in master_df.columns}
                formatted_rows.append(spacer)
            
            formatted_rows.append(row.to_dict())
            prev_comp = curr_comp
            
        master_df = pd.DataFrame(formatted_rows)
        # Ensure column order is preserved after dict conversion
        master_df = master_df[first_cols + other_cols]
        
        print(master_df.head())
        master_df.to_csv(output_file, index=False)
        print(f"\nSaved consolidated summaries to: {output_file}")
    else:
        print("Nothing to merge.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Root search directory")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    
    consolidate_nested_cv(args.dir, args.out)
