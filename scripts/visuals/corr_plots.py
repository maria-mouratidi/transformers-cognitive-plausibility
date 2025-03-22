import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_regplots(human_df, model_values, features, layer_idx, save_dir=None):
    for feature_name in features:
        plt.figure(figsize=(6, 5))
        sns.regplot(x=model_values, y=human_df[feature_name], scatter_kws={"s": 10}, line_kws={"color": "red"})
        plt.title(f'Regplot: Model Layer {layer_idx} Attention vs {feature_name}')
        plt.xlabel(f'Model Attention (Layer {layer_idx})')
        plt.ylabel(f'Human {feature_name}')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'regplot_{feature_name}.png')
            plt.savefig(save_path)
            print(f"Saved heatmap to {save_path}")

    plt.show()

def plot_corr_heatmap(human_df, model_values, features, layer_idx, save_dir=None):
    combined_df = human_df[features].copy()
    combined_df[f'Model_Layer_{layer_idx}'] = model_values

    corr_matrix = combined_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap (Human Features + Model Layer {layer_idx} Attention)")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'corr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    
    plt.show()
