import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_corr(results_df, pca_results_df, model_name, save_dir=None, significance_threshold=0.05):
    """
    Plot a heatmap combining feature and PCA component correlations.
    """
    # Reorder columns to put PC1 first
    def reorder_columns_pc1_first(df):
        cols = df.columns.tolist()
        if 'PC1' in cols:
            cols.remove('PC1')
            cols = ['PC1'] + cols
            return df[cols]
        return df
    
    method = 'spearman'  # or 'pearson', depending on your analysis
    results_df = results_df.copy()
    results_df['type'] = 'Feature'
    results_df = results_df.rename(columns={'feature': 'name'})
    pca_results_df = pca_results_df.copy()
    pca_results_df['type'] = 'PC'
    pca_results_df = pca_results_df.rename(columns={'principal_component': 'name'})
    combined = pd.concat([results_df, pca_results_df], ignore_index=True)

    # Exclude 'raw' attention method from combined heatmap
    flow_saliency_results_df = combined[combined['attn_method'] != 'raw'].copy()
    flow_saliency_results_df.loc[:, 'attn_layer'] = flow_saliency_results_df['attn_method']
    pivot = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='name', values=f'{method}_r')
    pvals = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='name', values=f'{method}_p_value')
    mask = pvals >= significance_threshold
    pivot = reorder_columns_pc1_first(pivot)
    pvals = reorder_columns_pc1_first(pvals)

    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
    plt.title(f"Model vs Eye-Gaze Correlations (p<{significance_threshold})")
    plt.xlabel('Eye-Gaze')
    plt.ylabel('Probing Method & Task')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"combined_corrs_{model_name}.png"))
        plt.close()

    # Plot task2 and task3 side by side with shared colorbar
    raw_results_task2 = combined[(combined['attn_method'] == 'raw') & (combined['task'] == 'task2')]
    raw_results_task3 = combined[(combined['attn_method'] == 'raw') & (combined['task'] == 'task3')]
    
    pivot_task2 = raw_results_task2.pivot(index='layer', columns='name', values=f'{method}_r')
    pvals_task2 = raw_results_task2.pivot(index='layer', columns='name', values=f'{method}_p_value')
    mask_task2 = pvals_task2 >= significance_threshold
    
    pivot_task3 = raw_results_task3.pivot(index='layer', columns='name', values=f'{method}_r')
    pvals_task3 = raw_results_task3.pivot(index='layer', columns='name', values=f'{method}_p_value')
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Task2 heatmap (left)
    sns.heatmap(pivot_task2, mask=mask_task2, annot=True, fmt=".2f", 
                cmap="coolwarm", center=0, ax=ax1, cbar=False,
                vmin=vmin, vmax=vmax)
    ax1.set_title(f"Task2")
    ax1.set_xlabel('Eye-Gaze Features')
    ax1.set_ylabel('Layer Attention')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Task3 heatmap (right)
    im = sns.heatmap(pivot_task3, mask=mask_task3, annot=True, fmt=".2f", 
                     cmap="coolwarm", center=0, ax=ax2, cbar=False,
                     vmin=vmin, vmax=vmax)
    ax2.set_title(f"Task3")
    ax2.set_xlabel('Eye-Gaze Features')
    ax2.set_ylabel('')  # Remove y-label for cleaner look
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    # Add shared colorbar
    cbar = fig.colorbar(im.collections[0], ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label(f'{method.title()} r', rotation=270, labelpad=20)
    
    plt.suptitle(f"Raw Attention Correlations ({method.title()}, p<{significance_threshold})", 
                 fontsize=14, y=0.98)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"raw_corrs_{model_name}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # # Optional: Still create individual plots if needed
    # for task in ['task2', 'task3']:
    #     raw_results_df = combined[(combined['attn_method'] == 'raw') & (combined['task'] == task)]
    #     pivot = raw_results_df.pivot(index='layer', columns='name', values=f'{method}_r')
    #     pvals = raw_results_df.pivot(index='layer', columns='name', values=f'{method}_p_value')
        
    #     # Reorder columns to put PC1 first
    #     pivot = reorder_columns_pc1_first(pivot)
    #     pvals = reorder_columns_pc1_first(pvals)
    #     mask = pvals >= significance_threshold

    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
    #                cbar_kws={'label': f'{method.title()} r'})
    #     plt.title(f"Model vs Eye-Gaze Correlations ({method.title()}, p<{significance_threshold})")
    #     plt.xlabel('Eye-Gaze Features')
    #     plt.ylabel('Layer Attention')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()

    #     if save_dir:
    #         os.makedirs(save_dir, exist_ok=True)
    #         plt.savefig(os.path.join(save_dir, f"raw_corrs_{task}_{model_name}.png"))
    #         plt.close()


def plot_gaze_intercorr(human_df, pca_df, features, save_dir=None):
    """
    Plot heatmap of inter-correlations among gaze features and PCA components.
    """
    combined = pd.concat([human_df[features], pca_df], axis=1)
    corr_matrix = combined.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'intercorrs.png')
        plt.savefig(save_path)
        print(f"Saved inter-correlation heatmap to {save_path}")


