import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
            print(f"Saved heatmap to {save_path}")

# In your `corr_plots.py`
def plot_feature_corr(results_df, attn_method, method='spearman', pca=False, save_dir=None, significance_threshold=0.05):

    col = 'principal_component' if pca else 'feature'
    pivot = results_df.pivot(index='layer', columns=col, values=f'{method}_r')
    pvals = results_df.pivot(index='layer', columns=col, values=f'{method}_p_value')

    # Create mask for non-significant p-values
    mask = pvals >= significance_threshold

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': f'{method.title()} r'})
    plt.title(f"Model vs Eye-Gaze Correlations ({method.title()}, p<{significance_threshold})")
    plt.xlabel('Eye-Gaze Features' if not pca else 'Eye-Gaze Principal Components')
    plt.ylabel('Layer Attention' if attn_method == "raw" else 'Attention Flow' if attn_method == "flow" else 'Gradient Saliency')
    plt.tight_layout()

    filename = f"pca_corr_{method}.png" if pca else f"corr_{method}.png"  
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{filename}")


def plot_eyegaze_corr(human_df, features, save_dir=None):
    """
    Heatmap of correlations among human eye-tracking features.
    """
    corr_matrix = human_df[features].corr(method='pearson')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0)
    plt.title("Eye-gaze Feature Inter-correlations")
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, f'intercorr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")


