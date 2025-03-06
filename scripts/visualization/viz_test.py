import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

save_dir = "/scratch/7982399/thesis/outputs/attention_plots"

def plot_attention_heatmaps(attention_tensor, sequence, batch_idx=0, save_dir=None, layers_to_plot=None):
    """
    Plot heatmaps for selected layers' attention weights for a given sentence in the batch and optionally save them.
    
    Args:
        attention_tensor: Tensor of shape [num_layers, batch_size, sent_len]
        sequence: List of words corresponding to the sentence
        batch_idx: Index of the batch to visualize
        save_dir: Directory to save the heatmaps (optional)
        layers_to_plot: List of layer indices to plot (optional)
    """
    num_layers, batch_size, sent_len = attention_tensor.shape
    assert batch_idx < batch_size, f"Batch index {batch_idx} out of range for batch size {batch_size}"
    assert len(sequence) == sent_len, f"Sequence length {len(sequence)} does not match sentence length {sent_len}"
    
    if layers_to_plot is None:
        layers_to_plot = [0, num_layers // 2, num_layers - 1]  # Default to first, middle, last layers
    
    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(sent_len, len(layers_to_plot) * 1.5), constrained_layout=True)
    if len(layers_to_plot) == 1:
        axes = [axes]
    
    for i, layer in enumerate(layers_to_plot):
        attention_weights = attention_tensor[layer, batch_idx, :]
        attention_weights = attention_weights.reshape(1, -1)  # Make it 2D for sns.heatmap
        sns.heatmap(attention_weights, ax=axes[i], cmap='Reds', cbar=True, xticklabels=sequence, yticklabels=[f'Layer {layer+1}'], linewidths=0.5, linecolor='black')
        axes[i].set_xlabel('Word Position')

    plt.suptitle(f'Attention Weights for Batch {batch_idx}', fontsize=14)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'attention_batch_{batch_idx}.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    
    plt.show()

attention_tensor = torch.load("/scratch/7982399/thesis/outputs/attention_tensor.pt")  # Load the attention tensor
num_layers, batch_size, sent_len = attention_tensor.shape
sentences = [
        "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
        "She won a Novel Prize in 1911.",
    ]
    
for batch_idx in range(batch_size):
    plot_attention_heatmaps(attention_tensor, sentences[batch_idx], batch_idx=batch_idx, save_dir="/scratch/7982399/thesis/outputs/attention_plots")