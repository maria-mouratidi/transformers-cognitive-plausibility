import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hist_kde_box(human_df, model_values, features, layer_idx, save_dir=None):
    for feature_name in features:
        plt.figure(figsize=(15, 5))
        
        # Histogram + KDE (Human)
        plt.subplot(1, 3, 1)
        sns.histplot(human_df[feature_name], kde=True, bins=30)
        plt.title(f'Eye-gaze Data - Histogram + KDE: {feature_name}')
        
        # Histogram + KDE (Model)
        plt.subplot(1, 3, 2)
        sns.histplot(model_values, kde=True, bins=30)
        plt.title(f'Model Data (Layer {layer_idx}) - Histogram + KDE')
        
        # Boxplots
        plt.subplot(1, 3, 3)
        sns.boxplot(data=[human_df[feature_name], model_values], orient='h')
        plt.yticks([0, 1], ['Eye-gaze', f'Model Layer {layer_idx}'])
        plt.title(f'Boxplots - Eye-gaze vs Model (Layer {layer_idx})')
        
        plt.tight_layout()
        
        if save_dir:
            # Modify save_dir to include a subdirectory for the layer index
            layer_save_dir = os.path.join(save_dir, f'layer_{layer_idx}')
            os.makedirs(layer_save_dir, exist_ok=True)
            save_path = os.path.join(layer_save_dir, f'hist_{feature_name}.png')
            plt.savefig(save_path)
            print(f"Saved heatmap to {save_path}")
    
    plt.show()