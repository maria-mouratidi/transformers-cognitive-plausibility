import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_corr(results_df, pca_results_df, model_name, save_dir=None, significance_threshold=0.05):
    """
    Plot a heatmap combining feature and PCA component correlations.
    """
    method = 'spearman'  # or 'pearson', depending on your analysis
    results_df = results_df.copy()
    results_df['type'] = 'Feature'
    results_df = results_df.rename(columns={'feature': 'name'})
    pca_results_df = pca_results_df.copy()
    pca_results_df['type'] = 'PC'
    pca_results_df = pca_results_df.rename(columns={'principal_component': 'name'})
    combined = pd.concat([results_df, pca_results_df], ignore_index=True)

    # Exclude 'raw' attention method from combined heatmap
    flow_saliency_results_df = combined[combined['attn_method'] != 'raw']

    # MultiIndex: (attn_method, task)
    flow_saliency_results_df['attn_layer'] = list(flow_saliency_results_df['attn_method'])
    pivot = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='name', values=f'{method}_r')
    pvals = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='name', values=f'{method}_p_value')
    mask = pvals >= significance_threshold

    plt.figure(figsize=(16, max(8, len(pivot))))
    sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
    plt.title(f"Model vs Eye-Gaze Correlations (p<{significance_threshold})")
    plt.xlabel('Eye-Gaze')
    plt.ylabel('Probing Method & Task')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"combined_corrs_{model_name}.png"))
        plt.close()

    #col = 'principal_component' if pca else 'feature'
    for task in ['task2', 'task3']:
        raw_results_df = combined[(combined['attn_method'] == 'raw') & (combined['task'] == task)]
        pivot = raw_results_df.pivot(index='layer', columns='name', values=f'{method}_r')
        pvals = raw_results_df.pivot(index='layer', columns='name', values=f'{method}_p_value')

        # Create mask for non-significant p-values
        mask = pvals >= significance_threshold

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
        plt.title(f"Model vs Eye-Gaze Correlations ({method.title()}, p<{significance_threshold})")
        plt.xlabel('Eye-Gaze Features')
        plt.ylabel('Layer Attention')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"raw_corrs_{task}_{model_name}.png"))
            plt.close()



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


