
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from ML_analysis.loading.config import ConfigLoader
from preprocessing.processflat import x_features_return
from ML_analysis.ml_utils import run_umap
from ML_analysis.analysis.plotting import plot_ols_diagnostics
from ML_analysis.analysis.regression import remove_missing_values

# === CONFIGURATION ===
PAPER_OUTPUT_DIR = "results/ML/paper_plots/"
FONT_SCALE = 1.3

def setup_plot_style():
    sns.set_style("white")
    sns.set_context("paper", font_scale=FONT_SCALE)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

def get_data_and_stats(target_col, args, df_input, df_meta):
    """
    Runs the regression logic for a target and returns everything needed for plotting.
    """
    # 1. Clean data (reusing same logic as regression.py)
    df_clean = remove_missing_values(df_input, df_meta, target_col)
    
    # 2. Merge features
    # df_merged includes ID, Group, Target, etc. + Voxels (X)
    df_merged, x_feature = x_features_return(df_clean, df_meta)
    
    # Check for NaNs/Strings in x_feature (robustness)
    if hasattr(x_feature, 'select_dtypes'):
        x_feature = x_feature.select_dtypes(include=[np.number]).fillna(0)
    
    # 3. UMAP
    # Ideally reuse existing embeddings if possible to match exactly, but re-run for now
    if args["umap"]:
        x_umap = run_umap(x_feature.values, random_state=42)
        x_ols = pd.DataFrame(x_umap, columns=['UMAP1', 'UMAP2'])
    else:
        x_ols = x_feature.copy()
    
    x_ols = sm.add_constant(x_ols)
    
    # 4. Target
    y = df_merged[target_col] # No Log transform for plotting
    
    # 5. Model
    model = sm.OLS(y, x_ols).fit()
    preds = model.predict(x_ols)
    residuals = y - preds
    
    # 6. Group Labels
    group_labels = df_merged['Group']
    
    # Stats
    stats = (model.rsquared, model.f_pvalue)
    
    return y, preds, residuals, group_labels, stats

def main():
    setup_plot_style()
    os.makedirs(PAPER_OUTPUT_DIR, exist_ok=True)
    
    loader = ConfigLoader()
    args, df_input, df_meta = loader.load_all()
    
    # Force params
    args["umap"] = True
    args["covariates"] = None
    
    # Prepare Figure: 2 Rows, 1 Column
    # Using 7x12 to allow 2 square-ish plots stacked
    fig, axes = plt.subplots(2, 1, figsize=(7, 12)) 
    
    # ROW 1: CDR_SB
    print("Processing CDR_SB...")
    y_cdr, pred_cdr, res_cdr, grp_cdr, stats_cdr = get_data_and_stats("CDR_SB", args, df_input, df_meta)
    plot_ols_diagnostics(
        y_cdr, pred_cdr, res_cdr, "CDR_SB", 
        plot_flag=False, save_flag=False, 
        color_by_group=True, group_labels=grp_cdr, 
        stats=stats_cdr, ax=axes[0]
    )
    # Custom Title or Y-label adjustments if needed
    axes[0].set_ylabel("Predicted Score (CDR)", fontweight='bold', fontsize=16)
    axes[0].set_title("CDR_SB", fontweight='bold', fontsize=18, loc='left', pad=10)
    
    # ROW 2: MMSE
    print("Processing MMSE...")
    y_mmse, pred_mmse, res_mmse, grp_mmse, stats_mmse = get_data_and_stats("MMSE", args, df_input, df_meta)
    plot_ols_diagnostics(
        y_mmse, pred_mmse, res_mmse, "MMSE", 
        plot_flag=False, save_flag=False, 
        color_by_group=True, group_labels=grp_mmse, 
        stats=stats_mmse, ax=axes[1]
    )
    axes[1].set_ylabel("Predicted Score (MMSE)", fontweight='bold', fontsize=16)
    axes[1].set_title("MMSE", fontweight='bold', fontsize=18, loc='left', pad=10)

    plt.tight_layout()
    
    # Save
    out_file = os.path.join(PAPER_OUTPUT_DIR, "combined_regression_diagnostics.png")
    out_svg = os.path.join(PAPER_OUTPUT_DIR, "combined_regression_diagnostics.svg")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"Saved combined plot to {out_file}")

if __name__ == "__main__":
    main()
