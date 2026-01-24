import pandas as pd
import os
import argparse
import sys
import json

# Define base paths - assume script is run from project root
ROOT_DIR = "results/ML"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_total_summary(ds_path):
    """Load the consolidated nested CV summary for the dataset."""
    path = os.path.join(ds_path, "total_nested_cv_summary.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except:
        return None

def get_val_from_total(df, comparison, model_name, metric):
    """
    Extract mean/std from the flat consolidated dataframe.
    Columns are like 'accuracy_mean', 'accuracy_std'.
    Rows are identified by 'comparison' and 'model'.
    """
    if df is None:
        return None, None
    
    # Normalize comparison naming
    # The consolidated file likely uses the folder name derived style (e.g. ad_psp).
    # We should match 'comparison' column. 
    # Usually consolidated script put 'ad_cbs' etc.
    # The 'comparison' arg comes from config, e.g. 'AD_PSP'.
    
    # Try exact match, then lower case, then underscore swap
    possible_comps = [comparison, comparison.lower(), f"{comparison.lower().split('_')[1]}_{comparison.lower().split('_')[0]}"]
    
    subset = pd.DataFrame()
    for pc in possible_comps:
        subset = df[(df["comparison"] == pc) & (df["model"] == model_name)]
        if not subset.empty:
            break
            
    if subset.empty:
        return None, None
        
    try:
        mean = subset[f"{metric}_mean"].iloc[0]
        std = subset[f"{metric}_std"].iloc[0]
        return mean, std
    except KeyError:
        # Metric might not exist
        return None, None

def load_p_value(ds_path, resolved_comp_dir, model_name):
    """
    Look for permutation_stats.csv in .../resolved_comp_dir/model_name/
    """
    # NOTE: Since we are moving to consolidated flows, we might want to read from total_permutation_stats.csv too?
    # User didn't explicitly ask to delete individual permutation files yet, only nested_cv_summary.
    # But for robustness, let's keep checking the individual file for now, 
    # OR better: read from total_permutation_stats.csv if available!
    
    total_perm_path = os.path.join(ds_path, "total_permutation_stats.csv")
    if os.path.exists(total_perm_path):
        try:
            df = pd.read_csv(total_perm_path)
            # Filter
            # Need to match 'groups' column (e.g. ADvsPSP or ad_psp, depending on how it was saved)
            # The consolidate script used 'groups' column.
            # Let's try to match.
            
            # Construct possible group names
            # 'resolved_comp_dir' passed here is usually 'ad_psp'
            # But the 'groups' col in perm file might be 'ADvsPSP'
            
            # Let's filter loosely or exactly
            # We don't have resolved_comp_dir easily if we are just looping config comparisons string.
            # Let's pass the config comparison string to this function instead of resolved dir.
            pass 
        except:
            pass
            
    # Fallback to individual file
    path = os.path.join(ds_path, resolved_comp_dir, model_name, "permutation_stats.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "p_value" in df.columns:
                return df["p_value"].iloc[0]
        except:
            pass
    return None

def report_formatted(mean, std, format_type):
    if mean is None:
        return "N/A"
    
    if format_type == "latex":
        return f"{mean:.3f} $\\pm$ {std:.3f}"
    elif format_type == "csv" or format_type == "tsv":
        return f"{mean:.3f} ± {std:.3f}"
    elif format_type == "markdown":
        return f"{mean:.3f} ± {std:.3f}"
    return "N/A"

def generate_report(args):
    # Load Configuration
    config = load_config(args.config)
    datasets = config["datasets"]
    comparisons = config["comparisons"]
    models_sorted = config["models"]
    model_map = config["model_display_names"]
    metrics = config["metrics"]
    include_p = config.get("include_p_value", False)
    
    if include_p and "p_value" not in metrics:
        metrics.append("p_value")

    # OUTPUT GENERATION
    output_lines = []
    
    # HEADER GENERATION (Latex/Markdown/TSV) - Same as before
    if args.format == "latex":
        output_lines.append(r"\begin{table}[h!]")
        output_lines.append(r"\centering")
        output_lines.append(r"\caption{Supervised ML classification performance (Mean $\pm$ SD).}")
        output_lines.append(r"\label{tab:ml_results}")
        output_lines.append(r"\resizebox{\textwidth}{!}{%")
        col_def = "ll" + "c" * len(metrics)
        output_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
        output_lines.append(r"\hline")
        headers = [m.replace("_", "-").title() for m in metrics]
        headers = ["AUC-ROC" if h == "Auc-Roc" else h for h in headers]
        headers = ["P-Value" if h == "P-Value" else h for h in headers]
        header_str = " & ".join([f"\\textbf{{{h}}}" for h in ["Comparison", "Model"] + headers])
        output_lines.append(f"{header_str} \\\\")
        output_lines.append(r"\hline")
    elif args.format == "markdown":
        headers = [m.replace("_", "-").title() for m in metrics]
        headers = ["AUC-ROC" if h == "Auc-Roc" else h for h in headers]
        headers = ["P-Value" if h == "P-Value" else h for h in headers]
        header = "| Comparison | Model | " + " | ".join(headers) + " |"
        sep = "| :--- | :--- | " + " | ".join([":---"] * len(metrics)) + " |"
        output_lines.append(header)
        output_lines.append(sep)
    elif args.format == "tsv":
        headers = [m.replace("_", "-").title() for m in metrics]
        header = "Comparison\tModel\t" + "\t".join(headers)
        output_lines.append(header)

    # DATA ITERATION
    for ds_info in datasets:
        ds_name = ds_info["name"]
        ds_path = ds_info["path"]
        
        # Load Total Summary CSV
        df_total = load_total_summary(ds_path)

        # Section Headers
        if args.format == "latex":
            output_lines.append(r"\multicolumn{" + str(len(metrics)+2) + r"}{c}{\textbf{" + ds_name + r"}} \\")
            output_lines.append(r"\hline")
        elif args.format == "markdown":
            output_lines.append(f"| **{ds_name}** |" + "|" * (len(metrics)+1))
        elif args.format == "tsv":
            output_lines.append(f"{ds_name}\t" + "\t" * (len(metrics)+1))

        for comp in comparisons:
            comp_lower = comp.lower()
            
            # Resolve directory just for P-Value lookup (since p-values still scattered / or partially consolidated)
            # The 'df_total' handles the main metrics.
            # We still need to guess the p-value location if not using total perm file.
            resolved_comp_dir = f"{comp_lower.split('_')[1]}_{comp_lower.split('_')[0]}" 
            # Try checking specific existence
            if not os.path.exists(os.path.join(ds_path, resolved_comp_dir)):
                 # try simple lower
                 resolved_comp_dir = comp_lower
            
            # Find best model
            best_model_name = None
            best_acc_val = -1.0
            if df_total is not None:
                for m in models_sorted:
                    acc, _ = get_val_from_total(df_total, comp, m, "accuracy")
                    if acc is not None and acc > best_acc_val:
                        best_acc_val = acc
                        best_model_name = m
            
            first_row = True
            for model in models_sorted:
                m_disp = model_map.get(model, model)
                comp_disp = ""
                if first_row:
                    if args.format == "latex":
                        comp_disp = f"\\textbf{{{comp.replace('_', ' vs ')}}}"
                    elif args.format == "markdown":
                        comp_disp = f"**{comp.replace('_', ' vs ')}**"
                    else:
                        comp_disp = comp.replace('_', ' vs ')
                
                row_cells = []
                
                # LOAD P-VALUE
                p_val_num = None
                if include_p:
                    p_val_num = load_p_value(ds_path, resolved_comp_dir, model)
                
                if df_total is None:
                    row_cells = ["Pending"] * len(metrics)
                else:
                    for met in metrics:
                        cell_str = "N/A"
                        if met == "p_value":
                            if p_val_num is not None:
                                if p_val_num < 0.001:
                                    cell_str = "< .001"
                                else:
                                    cell_str = f"{p_val_num:.3f}"
                            else:
                                cell_str = "-"
                        else:
                            mean, std = get_val_from_total(df_total, comp, model, met)
                            cell_str = report_formatted(mean, std, args.format)
                        
                        # Apply Bolding
                        if met == "accuracy" and model == best_model_name and met != "p_value":
                             if args.format == "latex":
                                 cell_str = f"\\textbf{{{cell_str}}}"
                             elif args.format == "markdown":
                                 cell_str = f"**{cell_str}**"
                        
                        row_cells.append(cell_str)
                
                if args.format == "latex":
                    data = " & ".join(row_cells)
                    output_lines.append(f"{comp_disp} & {m_disp} & {data} \\\\")
                elif args.format == "markdown":
                    data = " | ".join(row_cells)
                    output_lines.append(f"| {comp_disp} | {m_disp} | {data} |")
                elif args.format == "tsv":
                    data = "\t".join(row_cells)
                    output_lines.append(f"{comp_disp}\t{m_disp}\t{data}")
                
                first_row = False
            
            if args.format == "latex":
                output_lines.append(r"\hline")

    # FOOTER
    if args.format == "latex":
        output_lines.append(r"\end{tabular}%")
        output_lines.append(r"}")
        output_lines.append(r"\end{table}")
    
    # PRINT OR SAVE
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Report saved to {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML Result Tables")
    parser.add_argument("--format", choices=["latex", "markdown", "tsv"], default="markdown", help="Output format")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--config", default="src/ML_analysis/config/report_config.json", help="Path to config file")
    
    args = parser.parse_args()
    generate_report(args)
