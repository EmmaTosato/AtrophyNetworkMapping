
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ML_analysis.loading.config import ConfigLoader
from preprocessing.processflat import x_features_return
from ML_analysis.ml_utils import run_umap

# === CONFIGURATION ===
PAPER_OUTPUT_DIR = "results/ML/paper_plots/"
TARGETS = ["CDR_SB", "MMSE"]
GROUPS_ORDER = ["AD", "CBS", "PSP"]
SCATTER_KWS = dict(s=60, alpha=0.8, edgecolor="black", linewidth=0.5)
LINE_KWS = dict(linewidth=2.5)
FONT_SCALE = 1.3
PALETTE = "Set2"

def setup_plot_style():
    sns.set_style("white")
    sns.set_context("paper", font_scale=FONT_SCALE)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

def get_regression_data(target_col, args, df_input, df_meta):
    """
    Extracts x (UMAP features) and y (target) for a specific target variable.
    """
    # 1. Remove missing for THIS target
    # Align IDs first: intersection of input and metadata having target
    valid_ids = df_meta[df_meta[target_col].notna()]['ID']
    
    # Filter both DFs to this set of IDs to ensure clean alignment
    # (Assuming df_input is large/voxel data, filtering it is good)
    df_input_clean = df_input[df_input['ID'].isin(valid_ids)].copy()
    
    # Sort/Align by ID to ensure row-to-row correspondence if needed, 
    # though merge handles matching.
    
    # 2. Merge manually (safer than x_features_return)
    # We want X (voxels) and Y (Target) + Group
    # df_input has ID + Voxels
    # df_meta has ID + Group + Target + ...
    
    # Identify feature columns in df_input (all except ID)
    # Ensure they are numeric (floats/ints) to avoid strings like 'AD'
    feature_cols = [c for c in df_input_clean.columns if c != 'ID' and pd.api.types.is_numeric_dtype(df_input_clean[c])]
    
    # Fill NaNs with 0 (assuming masked voxel data)
    df_features = df_input_clean[feature_cols].fillna(0)
    x_feature = df_features.values
    
    # Check if empty
    if x_feature.shape[1] == 0:
        raise ValueError("No numeric feature columns found in input!")
    
    # Get metadata aligned to df_input_clean
    # We left merge meta onto input to keep input order/rows
    df_merged = pd.merge(df_input_clean[['ID']], df_meta, on='ID', how='left')
    
    # 3. UMAP
    if args["umap"]:
        x_umap = run_umap(x_feature, random_state=42)
    else:
        x_umap = x_feature
        
    # 4. Design Matrix
    x_ols = pd.DataFrame(x_umap, columns=['UMAP1', 'UMAP2'])
    x_ols = sm.add_constant(x_ols)
    
    # 5. Target
    y = df_merged[target_col]
    
    # 6. Model
    model = sm.OLS(y, x_ols).fit()
    preds = model.predict(x_ols)
    
    plot_df = pd.DataFrame({
        'True Score': y,
        'Predicted Score': preds,
        'Group': df_merged['Group']
    })
    
    return plot_df, model.rsquared, model.f_pvalue

def plot_paper_figure(df_cdr, stats_cdr, df_mmse, stats_mmse):
    """
    Generates the stacked/parallel plot matching the user image.
    Row 1: CDR (Left: All/Unfiltered?, Right: Filtered/Different?)
    Wait, the user image shows TWO plots per row? Left and Right? 
    Looking at the image/request: 
    "Plot paralleli con il main senza covariate ma sempre di gruppo"
    The user provided links to TWO images.
    - CDR_SB/.../no_threshold_diagnostics_diagnosis.png
    - MMSE/.../no_threshold_diagnostics_diagnosis.png
    
    The user wants these combined.
    Let's assume we want ONE column with 2 rows (CDR top, MMSE bottom) OR 2 columns side-by-side if they meant "parallel".
    
    Actually the user image upload (which I can't see but have description of) implies a layout.
    Text says "Prendi il seguente plot... E in ... Deve avere lo stesso layout. Qui i plot riportati nel paper."
    
    Common paper layout:
    A (CDR)
    B (MMSE)
    
    Let's make a vertical stack 2x1.
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=False)
    
    # --- CDR PLOT (Top) ---
    plot_single_regression(axes[0], df_cdr, stats_cdr, "CDR", show_legend=True)
    
    # --- MMSE PLOT (Bottom) ---
    plot_single_regression(axes[1], df_mmse, stats_mmse, "MMSE", show_legend=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_OUTPUT_DIR, "paper_regression_combined.png"), dpi=300)
    plt.savefig(os.path.join(PAPER_OUTPUT_DIR, "paper_regression_combined.svg"), format='svg')
    print(f"Saved combined plot to {PAPER_OUTPUT_DIR}")

def plot_single_regression(ax, df, stats, title_label, show_legend=False):
    r2, p = stats
    
    # Plot points & lines per group
    sns.scatterplot(
        data=df, x='True Score', y='Predicted Score', hue='Group', 
        palette=PALETTE, hue_order=GROUPS_ORDER, ax=ax, **SCATTER_KWS
    )
    
    # Add regression lines manually to ensure they match color
    # (sns.lmplot doesn't work well with existing axes in subplots for hue)
    colors = sns.color_palette(PALETTE, n_colors=len(GROUPS_ORDER))
    for i, grp in enumerate(GROUPS_ORDER):
        grp_data = df[df['Group'] == grp]
        if len(grp_data) > 1:
            sns.regplot(
                data=grp_data, x='True Score', y='Predicted Score', 
                color=colors[i], scatter=False, ci=95, ax=ax, truncate=False
            )

    # Styling
    ax.set_ylabel("Predicted Score", fontsize=14, fontweight='bold')
    ax.set_xlabel("True Score", fontsize=14, fontweight='bold')
    
    # Label (CDR/MMSE) on the left
    ax.text(-0.15, 0.5, title_label, transform=ax.transAxes, 
            fontsize=18, fontweight='bold', va='center', ha='right', rotation=0)

    # Stats annotation (Top Left inside)
    p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
    stats_text = f"$R^2 = {r2:.3f}$\n$p {p_text}$"
    ax.text(0.05, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top')
            
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Legend
    if show_legend:
        ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        ax.get_legend().remove()

def main():
    setup_plot_style()
    os.makedirs(PAPER_OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    loader = ConfigLoader()
    args, df_input, df_meta = loader.load_all()
    
    # Force settings for reproduction
    args["umap"] = True
    args["covariates"] = None # No covariates requested
    
    # Get Data for CDR
    print("Processing CDR_SB...")
    df_cdr, r2_cdr, p_cdr = get_regression_data("CDR_SB", args, df_input, df_meta)
    
    # Get Data for MMSE
    print("Processing MMSE...")
    df_mmse, r2_mmse, p_mmse = get_regression_data("MMSE", args, df_input, df_meta)
    
    # Plot
    plot_paper_figure(df_cdr, (r2_cdr, p_cdr), df_mmse, (r2_mmse, p_mmse))

if __name__ == "__main__":
    main()
