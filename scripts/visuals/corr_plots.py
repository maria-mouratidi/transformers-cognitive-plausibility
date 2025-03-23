import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_regplots(human_df, model_values, features, layer_idx, save_dir=None):
    for feature_name in features:
        plt.figure(figsize=(6, 5))
        sns.regplot(x=model_values, y=human_df[feature_name], scatter_kws={"s": 10}, line_kws={"color": "red"})
        plt.title(f'Regplot: Model Layer {layer_idx} Attention vs {feature_name}')
        plt.xlabel(f'Model Attention (Layer {layer_idx})')
        plt.ylabel(f'Eye-gaze: {feature_name}')
        plt.tight_layout()

        if save_dir:
            # Modify save_dir to include a subdirectory for the layer index
            layer_save_dir = os.path.join(save_dir, f'layer_{layer_idx}')
            os.makedirs(layer_save_dir, exist_ok=True)
            save_path = os.path.join(layer_save_dir, f'regplot_{feature_name}.png')
            plt.savefig(save_path)
            print(f"Saved heatmap to {save_path}")

    plt.show()

import numpy as np
from scipy.stats import pearsonr

def plot_feature_corr(results_df, corr_test='pearson', save_dir=None):
    """
    Heatmap of correlation values between layers and eye-gaze features.
    Non-significant correlations are greyed out.
    """
    # Pivot table for correlation values
    pivot_table = results_df.pivot(index='layer', columns='feature', values=f'{corr_test}_r')

    # Calculate p-values for significance testing
    p_values = results_df.pivot(index='layer', columns='feature', values=f'{corr_test}_p_value')

    # Create a mask for non-significant correlations
    mask = p_values >= 0.05

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", center=0, mask=mask, cbar_kws={'label': 'Correlation'})
    plt.title("Attention vs Eye-gaze Features Correlation (Layer-wise)")
    plt.ylabel("Model Layer")
    plt.xlabel("Eye-gaze Features")
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, f'corr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    plt.show()
    plt.close()


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
    
    plt.show()
    plt.close()

