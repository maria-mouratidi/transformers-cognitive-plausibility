import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hist_kde_box(human_df, model_values, features, layer_idx, save_dir=None):
    for feature_name in features:
        plt.figure(figsize=(15, 5))
        
        # Histogram + KDE (Human)
        plt.subplot(1, 3, 1)
        sns.histplot(human_df[feature_name], kde=True, bins=30)
        plt.title(f'Human Data - Histogram + KDE: {feature_name}')
        
        # Histogram + KDE (Model)
        plt.subplot(1, 3, 2)
        sns.histplot(model_values, kde=True, bins=30)
        plt.title(f'Model Data (Layer {layer_idx}) - Histogram + KDE')
        
        # Boxplots
        plt.subplot(1, 3, 3)
        sns.boxplot(data=[human_df[feature_name], model_values], orient='h')
        plt.yticks([0, 1], ['Human', f'Model Layer {layer_idx}'])
        plt.title(f'Boxplots - Human vs Model (Layer {layer_idx})')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'hist_kde_{feature_name}.png')
            plt.savefig(save_path)
            print(f"Saved heatmap to {save_path}")
    
    plt.show()
