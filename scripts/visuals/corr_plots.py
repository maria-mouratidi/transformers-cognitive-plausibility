import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_corr(results_df, pca_results_df, attn_method, method='spearman', save_dir=None, significance_threshold=0.05):
    """
    Plot a heatmap combining feature and PCA component correlations.
    """
    results_df = results_df.copy()
    results_df['type'] = 'Feature'
    results_df = results_df.rename(columns={'feature': 'name'})
    pca_results_df = pca_results_df.copy()
    pca_results_df['type'] = 'PC'
    pca_results_df = pca_results_df.rename(columns={'principal_component': 'name'})
    combined = pd.concat([results_df, pca_results_df], ignore_index=True)

    # Exclude 'raw' attention method from combined heatmap
    if 'attn_method' in combined.columns:
        combined = combined[combined['attn_method'] != 'raw']

        # MultiIndex: (attn_method, layer)
        combined['attn_layer'] = list(zip(combined['attn_method'], combined['layer']))
        pivot = combined.pivot(index=['task', 'attn_method', 'layer'], columns='name', values=f'{method}_r')
        pvals = combined.pivot(index=['task', 'attn_method', 'layer'], columns='name', values=f'{method}_p_value')
        mask = pvals >= significance_threshold

        plt.figure(figsize=(16, max(8, len(pivot))))
        sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
        plt.title(f"Model vs Eye-Gaze Correlations (Features & PCA, {method.title()}, p<{significance_threshold})")
        plt.xlabel('Eye-Gaze Features & Principal Components')
        plt.ylabel('Attention Method / Layer')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir)
            plt.close()
    else:
        # Add a column to distinguish features and PCs
        results_df = results_df.copy()
        results_df['type'] = 'Feature'
        results_df = results_df.rename(columns={'feature': 'name'})
        pca_results_df = pca_results_df.copy()
        pca_results_df['type'] = 'PC'
        pca_results_df = pca_results_df.rename(columns={'principal_component': 'name'})

        # Combine
        combined = pd.concat([results_df, pca_results_df], ignore_index=True)

        # Pivot for heatmap
        pivot = combined.pivot(index='layer', columns='name', values=f'{method}_r')
        pvals = combined.pivot(index='layer', columns='name', values=f'{method}_p_value')
        mask = pvals >= significance_threshold

        # Reorder columns: PCA components first, then features
        pc_names = pca_results_df['name'].unique().tolist()
        feature_names = results_df['name'].unique().tolist()
        ordered_columns = pc_names + [col for col in pivot.columns if col not in pc_names]
        pivot = pivot[ordered_columns]
        mask = mask[ordered_columns]

        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
        num_cols = len(pivot.columns)
        num_rows = len(pivot.index)
        plt.title(f"Model vs Eye-Gaze Correlations (Features & PCA, {method.title()}, p<{significance_threshold})")
        plt.xlabel('Eye-Gaze Features & Principal Components')
        plt.ylabel('Layer Attention' if attn_method == "raw" else 'Attention Flow' if attn_method == "flow" else 'Gradient Saliency')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir)
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


