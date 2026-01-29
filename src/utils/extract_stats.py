import os
import re

def parse_log_file(filepath):
    stats = {
        "R-squared": "N/A",
        "Adj. R-squared": "N/A",
        "Prob (F-statistic)": "N/A",
        "Permutation Line": "N/A"
    }
    
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Regex for OLS summary stats which share lines with other info
        # e.g. "Dep. Variable: CDR_SB   R-squared: 0.438"
        if "R-squared:" in line and "Adj. R-squared:" not in line:
            # Look for the last number in the line
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
                
        if "R^2 real:" in line and "shuffled mean:" in line:
            stats["Permutation Line"] = line
            
    return stats

def main():
    base_dir = "/data/users/etosato/ANM_Verona/results/ML"
    output_file = os.path.join(base_dir, "regression_stats_summary.txt")
    
    # Configuration
    cases = [
        {
            "name": "Caso NETWORK (results/ML/networks)",
            "base_path": os.path.join(base_dir, "networks/regression"),
            "targets": ["CDR_SB", "MMSE"], # CDR in user request maps to CDR_SB folder
            "cov_map": {"Covariates": "covariates", "No Covariates": "no_covariates"}
        },
        {
            "name": "Caso VOXEL (results/ML/voxel/umap_regression)",
            "base_path": os.path.join(base_dir, "voxel/umap_regression"),
            "targets": ["CDR_SB", "MMSE"],
            "cov_map": {"Covariates": "covariates", "No Covariates": "no_covariates"}
        }
    ]
    
    groups = ["AD", "CBS", "PSP"]
    
    with open(output_file, 'w') as out:
        for case in cases:
            out.write(f"{case['name']}\n")
            
            for target in case['targets']:
                display_target = "CDR" if target == "CDR_SB" else target
                out.write(f"- {display_target}\n")
                
                for cov_display, cov_folder in case['cov_map'].items():
                    out.write(f"\t- {cov_display}\n")
                    
                    for group in groups:
                        # Construct path
                        # Path: base / target / cov / group / group / log_no_threshold_Group_{group}.txt
                        # Wait, list_dir showed: regression/CDR_SB/covariates/group/AD
                        # So it is: base / target / cov / group / group_name
                        
                        log_dir = os.path.join(case['base_path'], target, cov_folder, "group", group)
                        log_file = os.path.join(log_dir, f"log_no_threshold_Group_{group}.txt")
                        
                        stats = parse_log_file(log_file)
                        
                        out.write(f"\t\t- {group}:\n")
                        if stats:
                            out.write(f"\t\t\tR-squared:                       {stats['R-squared']}\n")
                            out.write(f"\t\t\tAdj. R-squared:                  {stats['Adj. R-squared']}\n")
                            out.write(f"\t\t\tProb (F-statistic):              {stats['Prob (F-statistic)']}\n")
                            out.write(f"\t\t\t{stats['Permutation Line']}\n")
                        else:
                            out.write(f"\t\t\t(Log file not found: {log_file})\n")
    
    print(f"Summary written to {output_file}")
    with open(output_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
