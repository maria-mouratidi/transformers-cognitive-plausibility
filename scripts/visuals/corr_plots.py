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

def plot_layer_feature_corr(results_df, save_dir=None):
    """
    Heatmap of correlation values between layers and eye-gaze features.
    """
    pivot_table = results_df.pivot(index='layer', columns='feature', values='pearson_r')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", center=0)
    plt.title("Attention vs Eye-gaze Features Correlation (Layer-wise)")
    plt.ylabel("Transformer Layer")
    plt.xlabel("Eye-gaze Features")
    plt.tight_layout()


    if save_dir:
        save_path = os.path.join(save_dir, f'corr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    plt.show()


def plot_human_feature_corr(human_df, features, save_dir=None):
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

