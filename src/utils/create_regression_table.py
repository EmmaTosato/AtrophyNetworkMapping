import os
import re
import csv

def parse_log_file(filepath):
    stats = {
        "R-squared": "",
        "Adj. R-squared": "",
        "Prob (F-statistic)": ""
    }
    
    if not os.path.exists(filepath):
        return stats # Return empty strings
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        if "R-squared:" in line and "Adj. R-squared:" not in line:
            match = re.search(r"R-squared:\s+([0-9\.-]+)", line)
            if match:
                stats["R-squared"] = match.group(1)
                
        if "Adj. R-squared:" in line:
            match = re.search(r"Adj. R-squared:\s+([0-9\.-]+)", line)
            if match:
                stats["Adj. R-squared"] = match.group(1)
                
        if "Prob (F-statistic):" in line:
            match = re.search(r"Prob \(F-statistic\):\s+([0-9\.eE-]+)", line)
            if match:
                stats["Prob (F-statistic)"] = match.group(1)
            
    return stats

def main():
    base_dir = "/data/users/etosato/ANM_Verona/results/ML"
    output_file = os.path.join(base_dir, "regression_table.csv")
    
    # Configuration
    # We want Voxel first, then Network based on the image hint/logic
    datasets = [
        {
            "name": "VOXEL",
            "base_path": os.path.join(base_dir, "voxel/umap_regression"),
        },
        {
            "name": "NETWORK",
            "base_path": os.path.join(base_dir, "networks/regression"),
        }
    ]
    
    targets = ["CDR_SB", "MMSE"]
    target_display = ["CDR", "MMSE"]
    
    conditions = [
        ("Covariates", "covariates"),
        ("No Covariates", "no_covariates")
    ]
    
    groups = ["AD", "CBS", "PSP"]
    
    # Data gathering
    # Structure: data[dataset_name][condition_name][group][target] = stats
    data = {}
    
    for ds in datasets:
        ds_name = ds['name']
        data[ds_name] = {}
        for cond_disp, cond_folder in conditions:
            data[ds_name][cond_disp] = {}
            for group in groups:
                data[ds_name][cond_disp][group] = {}
                for target in targets:
                    # Path: base / target / cov / group / group / log...
                    # Wait, list_dir check from previous turn:
                    # results/ML/networks/regression/CDR_SB/covariates/group/AD/log...
                    log_dir = os.path.join(ds['base_path'], target, cond_folder, "group", group)
                    log_file = os.path.join(log_dir, f"log_no_threshold_Group_{group}.txt")
                    
                    data[ds_name][cond_disp][group][target] = parse_log_file(log_file)

    # Output to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for ds in datasets:
            ds_name = ds['name']
            
            # Title Row
            writer.writerow([ds_name])
            
            # Header Rows
            # Row 1: target spanning 3 columns each
            row1 = ["", ""] # 2 empty for row headers
            for t_disp in target_display:
                row1.append(t_disp)
                row1.append("")
                row1.append("")
            writer.writerow(row1)
            
            # Row 2: Metrics
            row2 = ["", ""]
            for _ in target_display:
                row2.extend(["R squared", "Adjusted R squared", "P - Value"])
            writer.writerow(row2)
            
            # Data Rows
            for cond_disp, _ in conditions:
                # Condition Header (e.g. Covariates) in first column? 
                # Or just listing AD/CBS/PSP.
                # Image structure suggests: 
                # Covariates - AD
                #            - CBS
                #            - PSP
                
                # First group row gets the Condition label
                first = True
                for group in groups:
                    row = []
                    if first:
                        row.append(cond_disp)
                        first = False
                    else:
                        row.append("")
                    
                    row.append(group)
                    
                    for target in targets:
                        stats = data[ds_name][cond_disp][group][target]
                        row.append(stats["R-squared"])
                        row.append(stats["Adj. R-squared"])
                        row.append(stats["Prob (F-statistic)"])
                    
                    writer.writerow(row)
                
            # Separator blank lines
            writer.writerow([])
            writer.writerow([])
            
    print(f"Table written to {output_file}")

if __name__ == "__main__":
    main()
