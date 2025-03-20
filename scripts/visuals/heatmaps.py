import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.probing.raw import subset

save_dir = "/scratch/7982399/thesis/outputs/attention_plots"

def plot_attention_heatmaps(attention_tensor, sentence, batch_idx, save_dir=None, layers_to_plot=None):
    """
    Plot heatmaps for selected layers' attention weights for a given sentence in the batch and optionally save them.
    
    Args:
        attention_tensor: Tensor of shape [num_layers, batch_size, sent_len]
        sequence: List of words corresponding to the sentence
        batch_idx: Index of the batch to visualize
        save_dir: Directory to save the heatmaps (optional)
        layers_to_plot: List of layer indices to plot (optional)
    """
    _, batch_size, att_len = attention_tensor.shape

    # Initial checks
    assert batch_idx < batch_size, f"Batch index {batch_idx} out of range for batch size {batch_size}"
    pad_area = attention_tensor[:, batch_idx, len(sentence)+1:]
    assert torch.all(pad_area == 0), "Non-zero values found in the assumed padded area: this likely means a mismatch between the sentence length and attention tensor's last dimension"
    
    if layers_to_plot is None:
        layers_to_plot = [0, num_layers // 2, num_layers - 1]  # Default to first, middle, last layers
    
    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(att_len, len(layers_to_plot) * 1.5), constrained_layout=True)
    if len(layers_to_plot) == 1:
        axes = [axes]
    
    for i, layer in enumerate(layers_to_plot):
        attention_weights = attention_tensor[layer, batch_idx, :len(sentence)]
        attention_weights = attention_weights.reshape(1, -1)  # Make it 2D for sns.heatmap
        sns.heatmap(attention_weights, ax=axes[i], cmap='Reds', cbar=True, xticklabels=sentence, yticklabels=[f'Layer {layer+1}'], linewidths=0.5, linecolor='black')
        axes[i].set_xlabel('Word Position')

    plt.suptitle(f'Attention Weights', fontsize=14)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_attn_sent{batch_idx}.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import json

    # Load processed attention tensor and sentences
    data = torch.load("/scratch/7982399/thesis/outputs/attention_processed.pt")  # Load the attention tensor
    
    # Extract each component
    attention = data['attention_processed']
    prompt_len = data['prompt_len']
    
    with open('materials/sentences.json', 'r') as f:
        sentences = json.load(f)

    sentences = sentences[:subset]
    num_layers, batch_size, max_sent_len = attention.shape
    #sentences = pad_lists(sentences, max_sent_len)
    # Loop through sentences in the batch and plot attention heatmaps
    for batch_idx in range(batch_size):
        plot_attention_heatmaps(attention, sentences[batch_idx], batch_idx=batch_idx, save_dir="/scratch/7982399/thesis/outputs/attention_plots")