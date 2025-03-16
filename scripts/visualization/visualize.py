import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.probing.raw import get_prompt_token_count
import os

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
    num_layers, batch_size, sent_len = attention_tensor.shape
    assert batch_idx < batch_size, f"Batch index {batch_idx} out of range for batch size {batch_size}"
    assert len(sentence) == sent_len, f"Sentence length {len(sentence)} does not match attention's last dim {sent_len}"
    
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
        save_path = os.path.join(save_dir, f'attention_processed_batch_{batch_idx}.png')
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    
    plt.show()

def pad_lists(lists, max_sent_len, pad_value=""):
    return [lst[:max_sent_len] + [pad_value] * (max_sent_len - len(lst)) for lst in lists]


if __name__ == "__main__":

    # Load processed attention tensor and sentences
    attention_tensor = torch.load("/scratch/7982399/thesis/outputs/attention_processed.pt")  # Load the attention tensor
    sentences = torch.load("/scratch/7982399/thesis/outputs/attention_data.pt")['sentences']
    
    num_layers, batch_size, max_sent_len = attention_tensor.shape
    prompt_word_count = get_prompt_token_count("task2")
    # Remove the prompt for proper plotting
    sentences = [sentence[prompt_word_count:] for sentence in sentences]  # Remove prompt words
    # Pad sentences to the same length for tensor alignment
    sentences = pad_lists(sentences, max_sent_len)

    # Loop through sentences in the batch and plot attention heatmaps
    for batch_idx in range(batch_size):
        plot_attention_heatmaps(attention_tensor, sentences[batch_idx], batch_idx=batch_idx, save_dir="/scratch/7982399/thesis/outputs/attention_plots")