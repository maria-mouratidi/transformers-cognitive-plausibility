import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def plot_regplots(human_df, model_values, features, layer_idx, attn_method, save_dir=None):
    for feature_name in features:
        plt.figure(figsize=(6, 5))
        
        sns.regplot(x=model_values, y=human_df[feature_name], scatter_kws={"s": 10}, line_kws={"color": "red"})
        plt.title(f'Model vs Eye-gaze')
        plt.xlabel(f'Attention Layer {layer_idx}') if attn_method == "raw" else plt.xlabel('Attention Flow') if attn_method == "flow" else plt.xlabel('Gradient Saliency')
        plt.ylabel(f'{feature_name}')
        plt.tight_layout()

        if save_dir:
            # Modify save_dir to include a subdirectory for the layer index
            layer_save_dir = os.path.join(save_dir, f'layer_{layer_idx}')
            os.makedirs(layer_save_dir, exist_ok=True)
            save_path = os.path.join(layer_save_dir, f'regplot_{feature_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved heatmap to {save_path}")

def plot_combined_corr(results_df, pca_results_df, attn_method, method='spearman', save_dir=None, significance_threshold=0.05):
    """
    Plot a heatmap combining feature and PCA component correlations.
    """
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
    plt.figure(figsize=(0.6*num_cols + 2, 0.5*num_rows + 2))
    plt.title(f"Model vs Eye-Gaze Correlations (Features & PCA, {method.title()}, p<{significance_threshold})")
    plt.xlabel('Eye-Gaze Features & Principal Components')
    plt.ylabel('Layer Attention' if attn_method == "raw" else 'Attention Flow' if attn_method == "flow" else 'Gradient Saliency')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/combined_corr_{method}.png")
        plt.close()
        print(f"Saved combined correlation heatmap to {save_dir}/combined_corr_{method}.png")

def plot_feature_intercorr(human_df, pca_df, features, save_dir=None):
    """
    Plot heatmap of inter-correlations among gaze features and PCA components.
    """
    combined = pd.concat([human_df[features], pca_df], axis=1)
    corr_matrix = combined.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0)
    plt.title("Inter-correlations: Eye-gaze Features & PCA Components")
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'feature_pca_intercorr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved inter-correlation heatmap to {save_path}")


