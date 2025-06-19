import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
palette = "RdBu_r"  #diverging color palette

def set_academic_rcparams():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    })

def plot_corr(results_df, pca_results_df, model_name, save_dir=None, significance_threshold=0.05):
    """
    Plot a heatmap combining feature and PCA component correlations.
    Optimized for academic paper publication.
    """
    set_academic_rcparams()
    
    # Reorder columns to put PC1 first
    def reorder_columns_pc1_first(df):
        cols = df.columns.tolist()
        if 'PC1' in cols:
            cols.remove('PC1')
            cols = ['PC1'] + cols
            return df[cols]
        return df
    
    method = 'spearman'  # or 'pearson'
    results_df = results_df.copy()
    pca_results_df = pca_results_df.copy()
    results_df['type'] = 'Feature'
    pca_results_df['type'] = 'PC'
    pca_results_df = pca_results_df.rename(columns={'principal_component': 'feature'})
    combined = pd.concat([results_df, pca_results_df], ignore_index=True)

    # Exclude 'raw' attention method from combined heatmap
    flow_saliency_results_df = combined[combined['attn_method'] != 'raw'].copy()
    flow_saliency_results_df.loc[:, 'attn_layer'] = flow_saliency_results_df['attn_method']
    pivot = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='feature', values=f'{method}_r')
    pvals = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='feature', values=f'{method}_p_value')
    mask = pvals >= significance_threshold
    pivot = reorder_columns_pc1_first(pivot)
    pvals = reorder_columns_pc1_first(pvals)

    # First plot: Combined correlations
    plt.figure(figsize=(7, 3.5))
    sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", center=0, 
                cmap=palette, cbar_kws={'label': f'{method.title()} r', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"combined_corrs_{model_name}.pdf"), 
                   dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

    # Plot task2 and task3 side by side with shared colorbar
    raw_results_task2 = combined[(combined['attn_method'] == 'raw') & (combined['task'] == 'task2')]
    raw_results_task3 = combined[(combined['attn_method'] == 'raw') & (combined['task'] == 'task3')]
    
    pivot_task2 = raw_results_task2.pivot(index='layer', columns='feature', values=f'{method}_r')
    pvals_task2 = raw_results_task2.pivot(index='layer', columns='feature', values=f'{method}_p_value')
    mask_task2 = pvals_task2 >= significance_threshold
    
    pivot_task3 = raw_results_task3.pivot(index='layer', columns='feature', values=f'{method}_r')
    pvals_task3 = raw_results_task3.pivot(index='layer', columns='feature', values=f'{method}_p_value')
    mask_task3 = pvals_task3 >= significance_threshold

    pivot_task2 = reorder_columns_pc1_first(pivot_task2)
    pvals_task2 = reorder_columns_pc1_first(pvals_task2)
    mask_task2 = pvals_task2 >= significance_threshold
    
    pivot_task3 = reorder_columns_pc1_first(pivot_task3)
    pvals_task3 = reorder_columns_pc1_first(pvals_task3)
    mask_task3 = pvals_task3 >= significance_threshold
    
    # Determine shared color scale
    vmin = min(pivot_task2.min().min(), pivot_task3.min().min())
    vmax = max(pivot_task2.max().max(), pivot_task3.max().max())
    
    # Create side-by-side plot
    if model_name == 'llama':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))  # More compact
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # Task2 heatmap (left)
    sns.heatmap(pivot_task2, mask=mask_task2, annot=True, fmt=".2f", 
                 cmap=palette, center=0, ax=ax1, cbar=False,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white')
    ax1.set_title("Task 2", fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Layer')
    ax1.tick_params(axis='y', rotation=0)
    
    # Task3 heatmap (right)
    im = sns.heatmap(pivot_task3, mask=mask_task3, annot=True, fmt=".2f", 
                     cmap=palette, center=0, ax=ax2, cbar=False,
                     vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='white')
    ax2.set_title("Task 3", fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')  # Remove y-label
    ax2.tick_params(axis='y', rotation=0)
    
    # Add shared colorbar
    cbar = fig.colorbar(im.collections[0], ax=[ax1, ax2], shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label(f'{method.title()} r', rotation=270, labelpad=15)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"raw_corrs_{model_name}.pdf"), 
                   dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

def plot_gaze_intercorr(human_df, pca_df, features, save_dir=None):
    """
    Plot heatmap of inter-correlations among gaze features and PCA components.
    """
    set_academic_rcparams()
    # Replace feature names for display
    combined = pd.concat([human_df[features], pca_df], axis=1)
    combined = combined.rename(columns={'meanPupilSize': 'mPS', 'nFixations': 'F'})
    corr_matrix = combined.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, center=0, cmap=palette)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'intercorrs.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved inter-correlation heatmap to {save_path}")


