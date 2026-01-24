import pandas as pd
import os
import glob
import argparse

def consolidate_permutations(base_dir="results/ML", output_file="results/ML/total_permutation_stats.csv"):
    print(f"Searching for permutation_stats.csv in {base_dir}...")
    
    # Recursive search
    # Only look for files named exactly 'permutation_stats.csv'
    files = glob.glob(os.path.join(base_dir, "**", "permutation_stats.csv"), recursive=True)
    
    if not files:
        print("No permutation files found.")
        return

    print(f"Found {len(files)} files. Merging...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Add metadata path if needed, though rows supposedly contain self-identifying info
            # We can ensure 'dataset', 'groups', 'model' are present. 
            # If the individual csv missed something, we could infer from path, but our script put them in.
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        
        # Sort for readability: Dataset -> Groups -> Model
        # We assume columns exist.
        # Round numeric columns
        numeric_cols = master_df.select_dtypes(include=['float', 'float64']).columns
        master_df[numeric_cols] = master_df[numeric_cols].round(4)

        # Sort by groups (comparison) and model
        target_cols = ["dataset", "groups", "model"]
        sort_cols = [c for c in target_cols if c in master_df.columns]
        if sort_cols:
            master_df.sort_values(by=sort_cols, inplace=True)
            
        # Insert spacing rows
        formatted_rows = []
        prev_group = None
        # Use 'groups' column for separation
        if "groups" in master_df.columns:
            for _, row in master_df.iterrows():
                curr_group = row['groups']
                if prev_group is not None and curr_group != prev_group:
                    spacer = {c: "" for c in master_df.columns}
                    formatted_rows.append(spacer)
                
                formatted_rows.append(row.to_dict())
                prev_group = curr_group
            
            master_df = pd.DataFrame(formatted_rows)
            # Ensure col order, putting standard ones first
            cols = list(master_df.columns)
            prio_cols = ["dataset", "groups", "model"]
            other_cols = [c for c in cols if c not in prio_cols]
            # Verify they exist
            final_prio = [c for c in prio_cols if c in cols]
            final_other = [c for c in cols if c not in final_prio]
            master_df = master_df[final_prio + final_other]

        print(master_df.head())
        master_df.to_csv(output_file, index=False)
        print(f"\nSaved consolidated results to: {output_file}")
    else:
        print("Nothing to merge.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/ML", help="Root search directory")
    parser.add_argument("--out", default="results/ML/total_permutation_stats.csv", help="Output CSV path")
    args = parser.parse_args()
    
    consolidate_permutations(args.dir, args.out)
