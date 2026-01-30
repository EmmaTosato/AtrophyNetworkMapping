# plotting.py

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.collections as mcoll
from sklearn.metrics import confusion_matrix

# === Utility Functions ===
def clean_title_string(title):
    title = re.sub(r'\bcovariates\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[\s\-]+', '_', title)
    title = re.sub(r'_+', '_', title)
    return title.strip('_').lower()

# Consistent color mapping for diagnostic groups (Set2 palette order)
GROUP_COLORS = {
    'AD': '#66c2a5',   # Turchese (Set2 index 0)
    'CBS': '#fc8d62',  # Arancione (Set2 index 1)
    'PSP': '#8da0cb',  # Violetto (Set2 index 2)
}
DEFAULT_COLOR = '#61bdcd'  # Fallback color

# === Regression & Diagnostic Plots ===
def plot_ols_diagnostics(target, predictions, residuals,
                         title, save_path=None,
                         plot_flag=True, save_flag=False,
                         color_by_group=False, group_labels=None,
                         stats=None, ax=None, group_name=None):
    """
    Plots OLS diagnostics (True vs Predicted) with optional group coloring.
    Uses sns.lmplot for grouped data to match original paper style.
    """
    def _format_axes(ax, xlabel="", ylabel="", fontsize=18):
        # User requested removals of axis labels
        if xlabel: ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold', labelpad=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.tick_params(labelsize=14, width=1.0)
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    # Logic: If color_by_group, use sns.lmplot (creates new Figure).
    # If standard, use ax if provided or create new.

    if color_by_group and group_labels is not None:
        df_plot = pd.DataFrame({
            'target': target,
            'predictions': predictions,
            'residuals': residuals,
            # Force using group_name if available to ensure palette match (e.g. 'AD')
            # Otherwise use group_labels (Series)
            'group': group_name if group_name else (group_labels if group_labels is not None else "Unknown")
        })
        
        # Determine axis limits for square aspect
        x_min, x_max = df_plot['target'].min(), df_plot['target'].max()
        y_min, y_max = df_plot['predictions'].min(), df_plot['predictions'].max()
        axis_min = min(x_min, y_min) - 1
        axis_max = max(x_max, y_max) + 1

        # Use lmplot matching the EXACT separate file style
        g = sns.lmplot(
            data=df_plot,
            x='target',
            y='predictions',
            hue='group',
            palette=GROUP_COLORS,
            height=6,
            aspect=1,
            scatter_kws=dict(s=110, alpha=0.9, edgecolor="black", linewidths=0.6),
            line_kws=dict(linewidth=2.0),
            ci=95,
            legend=False 
        )
        
        # Set uniform limits
        g.set(xlim=(axis_min, axis_max), ylim=(axis_min, axis_max))
        
        for ax_curr in g.axes.flat:
            ax_curr.set_aspect('equal', adjustable='box')
            _format_axes(ax_curr)
            # Fix transparency of confidence intervals
            for coll in ax_curr.collections:
                if isinstance(coll, mcoll.PolyCollection):
                    coll.set_alpha(0.2)
            
            # Add Stats if provided (DISABLED)
            # Add Stats if provided (DISABLED)
            # if stats:
            #     r2, p = stats
            #     p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
            #     stats_text = f"$R^2 = {r2:.3f}$\n$p {p_text}$"
            #     ax_curr.text(0.05, 0.90, stats_text, transform=ax_curr.transAxes, 
            #                  fontsize=16, fontweight='bold', va='top', ha='left')


        # Legend Styling
        # sns.lmplot places legend outside by default. We can customize it from g._legend
        # Legend Styling
        # sns.lmplot places legend outside by default. We can customize it from g._legend
        if g._legend:
             # Remove title or customize? User wants simple style.
             # If it's single group, maybe remove legend entirely?
             # For 'labelled' plot (all groups), legend is useful.
             # For single group, redundant if we know the group.
             if group_name: 
                 g._legend.remove() # Remove legend for single group plots
             else:
                 g._legend.set_title("Group")
                 g._legend.get_title().set_fontsize(16)
                 g._legend.get_title().set_fontweight('bold')
                 for t in g._legend.texts:
                     t.set_fontsize(16)
                 g._legend.get_frame().set_facecolor("white")
                 g._legend.get_frame().set_linewidth(0)
        
        if save_path and save_flag:
            clean_t = clean_title_string(title)
            # User request: for single groups (group_name present), use _diagnostics.png
            # For all groups combined, use _diagnostics_labelled.png
            suffix = "_diagnostics.png" if group_name else "_diagnostics_labelled.png"
            fname = clean_t + suffix
            g.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight', pad_inches=0.1)

        if plot_flag:
            plt.show()
        
        plt.close() # Close figure

    else:
        # SINGLE PLOT (No hues)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        df_plot = pd.DataFrame({'target': target, 'predictions': predictions})

        # Use group-specific color if provided
        plot_color = GROUP_COLORS.get(group_name, DEFAULT_COLOR) if group_name else DEFAULT_COLOR

        sns.scatterplot(
            data=df_plot, x='target', y='predictions',
            color=plot_color, edgecolor='black', alpha=0.9, s=110, linewidth=0.6, ax=ax
        )
        sns.regplot(
             data=df_plot, x='target', y='predictions',
             color=plot_color, scatter=False, ci=95, ax=ax, truncate=False,
             line_kws={'linewidth': 2.5}
        )

        # Stats
        if stats:
            r2, p = stats
            p_text = "< .001" if p < 0.001 else f"= {p:.3f}"
            stats_text = f"$R^2 = {r2:.3f}$\n$p {p_text}$"
            ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top', ha='left')

        _format_axes(ax)
        _format_axes(ax)
        # if not color_by_group: # Title REMOVED per user request
        #      ax.set_title("OLS True vs Predicted", fontsize=16, fontweight='bold', pad=10)
        
        if ax is not None and not created_fig:
            # If we are part of a subplot, we don't save or show individually here usually,
            # unless instructed. But existing 'regression.py' loop assumes we save.
            # But the 'else' block here is for single plots.
            pass

        if save_path and save_flag and created_fig:
            fname = clean_title_string(title) + "_diagnostics.png"
            fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight', pad_inches=0.1)

        if plot_flag and created_fig:
            plt.show()

        if created_fig:
            plt.close(fig)


def plot_actual_vs_predicted(target, predictions,
                             title, save_path=None,
                             plot_flag=False, save_flag=False):
    """
    Plots histograms of actual vs predicted values with consistent visual styling.
    """
    title_font = FontProperties(family="DejaVu Sans", weight='bold', size=20)
    label_font = FontProperties(family="DejaVu Sans", weight='bold', size=18)

    bins = np.arange(min(target), max(target) + 0.5, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axs[0].hist(target, bins=bins, color='#61bdcd', edgecolor='black', alpha=0.85)
    axs[0].set_title('Actual Distribution', fontproperties=title_font)
    axs[0].set_xlabel("Value", fontproperties=label_font)
    axs[0].set_ylabel("Count", fontproperties=label_font)

    axs[1].hist(predictions, bins=bins, color='#95d6bb', edgecolor='black', alpha=0.85)
    axs[1].set_title('Predicted Distribution', fontproperties=title_font)
    axs[1].set_xlabel("Value", fontproperties=label_font)
    axs[1].set_ylabel("Count", fontproperties=label_font)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.tick_params(labelsize=12)
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()

    if save_path and save_flag:
        filename = f"{clean_title_string(title)}_distribution.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()

    plt.close()


# === UMAP & Clustering Plots ===
def plot_clusters_vs_groups(x_umap, labels_dict, group_column,
                            save_path, title_prefix,
                            margin=2.0,
                            plot_flag=True, save_flag=False, title_flag=False,
                            colors_gmm=False, separated=False):
    """
    Plots clustering results side-by-side with group labels using UMAP coordinates.
    Also generates separate plots for cluster and group if 'separated' is True.
    """
    def _format_umap_axes(ax, xlabel="UMAP 1", ylabel="UMAP 2", fontsize=14):
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.tick_params(labelsize=14)
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    n = len(labels_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))
    if n == 1:
        axes = [axes]

    x1, x2 = x_umap[:, 0], x_umap[:, 1]
    min_val = min(x1.min(), x2.min()) - margin
    max_val = max(x1.max(), x2.max()) + margin

    left_colors = ['#E24141', '#74c476', '#7BD3EA', '#fd8d3c', '#37659e', '#fbbabd', '#ffdb24',
                   '#413d7b', '#9dd569', '#e84a9b', '#056c39', '#6788ee']
    right_colors = sns.color_palette("Set2")[2:] if colors_gmm else sns.color_palette("Set2")

    for i, (name, labels) in enumerate(labels_dict.items()):
        df_plot = pd.DataFrame({'X1': x1, 'X2': x2, 'cluster': labels, 'label': group_column}).dropna(subset=['label'])

        # Left subplot: by cluster
        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='cluster', palette=left_colors,
            s=80, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][0]
        )

        # Right subplot: by group
        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='label', palette=right_colors,
            s=80, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][1]
        )

        for ax_, title_ in zip(axes[i], ['Cluster', 'Label']):
            leg = ax_.get_legend()
            if leg is not None:
                leg.set_title(title_, prop={'weight': 'bold', 'size': 16})
                for text in leg.get_texts():
                    text.set_fontsize(16)
                    text.set_fontweight('regular')
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("black")
                leg.get_frame().set_linewidth(0)

        for ax in axes[i]:
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            _format_umap_axes(ax)

        if title_flag:
            axes[i][0].set_title(name, fontsize=16, fontweight='bold')
            axes[i][1].set_title(f"{name} - Labeling by {group_column.name}", fontsize=16, fontweight='bold')

        # ---------- Separate plots ----------
        if separated:
            for view, hue_col, palette, suffix in [("Cluster view", "cluster", left_colors, "cluster"),
                                                   ("Group view", "label", right_colors, "group")]:
                fig_sep, ax_sep = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=df_plot, x="X1", y="X2", hue=hue_col, palette=palette,
                    s=110, alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax_sep
                )

                leg = ax_sep.get_legend()
                if leg is not None:
                    leg.set_title(hue_col.capitalize(), prop={'weight': 'bold', 'size': 16})
                    for text in leg.get_texts():
                        text.set_fontsize(16)
                        text.set_fontweight('regular')
                    leg.get_frame().set_facecolor("white")
                    leg.get_frame().set_edgecolor("black")
                    leg.get_frame().set_linewidth(0)

                _format_umap_axes(ax_sep)
                if title_flag:
                    ax_sep.set_title(f"{name} - {view}", fontsize=16, fontweight='bold')
                if save_flag and save_path:
                    fname = clean_title_string(f"{title_prefix}_{name}_{suffix}") + ".png"
                    fig_sep.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
                if plot_flag:
                    plt.show()
                plt.close(fig_sep)

    # ----- Final combined plot -----
    if title_flag:
        fig.suptitle("Clustering Results", fontsize=22, fontweight='bold')
        fig.text(0.5, 0.92, title_prefix, fontsize=16, ha='center')

    if save_path and save_flag:
        os.makedirs(save_path, exist_ok=True)
        fname = clean_title_string(title_prefix) + "_clustering.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')

    if plot_flag:
        plt.show()

    plt.close(fig)



def plot_umap_embedding(labeling_umap,
                        title=None,
                        save_path=None,
                        plot_flag=True,
                        save_flag=False,
                        title_flag=False,
                        dot_color="#d74c4c",
                        ):
    """
    Plots 2D UMAP embedding using the 'labeling_umap' DataFrame.
    """
    clean_title = re.sub(r'[\s\-]+', '_', title.strip().lower()) if title else "umap"

    plt.figure(figsize=(6, 4))
    plt.scatter(
        labeling_umap["X1"], labeling_umap["X2"],
        s=80, alpha=0.9,
        color=dot_color,
        edgecolor='black',
        linewidth=0.5
    )
    if title_flag is not False:
        plt.title(f'UMAP Embedding - {title}', fontsize=14, fontweight='bold')
    plt.xlabel("UMAP 1", fontsize=12, fontweight='bold')
    plt.ylabel("UMAP 2", fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.grid(False)
    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    if save_path and save_flag:
        file_standard = os.path.join(save_path, f"{clean_title}_embedding.png")
        plt.savefig(file_standard, dpi=300, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()


# === Classification Evaluation ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Plots a stylized confusion matrix with enhanced readability and thick external border.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = sns.heatmap(
        cm, annot=True, fmt='d', cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor='black', ax=ax,
        annot_kws={"size": 20, "weight": "bold"},
    )

    # Axis labels with spacing
    ax.set_xlabel("Predicted", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel("True", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)

    # Enlarge colorbar tick labels
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')


    # Tick labels styling
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticklabels(class_names, fontsize=18, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=18, fontweight='bold', rotation=0)

    # Add THICK outer border as a rectangle
    rows, cols = cm.shape
    rect = patches.Rectangle(
        (0, 0), cols, rows,
        linewidth=4.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    plt.close()





