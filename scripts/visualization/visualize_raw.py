import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys

from scripts.probing.raw import *
from scripts.probing.load_model import *

def plot_attention_heatmap(attention_weights: torch.Tensor, 
                          sentences: List[str], 
                          save_path: str = None,
                          title: str = "Attention Weights"):
    """
    Plot attention heatmap for each sentence in the batch.
    """
    attention_np = attention_weights.cpu().numpy()
    
    # Handle different dimensionality
    if len(attention_np.shape) == 3:  # If attention_np is 3D (batch/layer, seq, seq)
        num_items = attention_np.shape[0]
    else:  # If attention_np is 2D (seq, seq)
        num_items = 1
        attention_np = attention_np[None, :]  # Add batch dimension
    
    fig, axes = plt.subplots(num_items, 1, 
                            figsize=(12, 4*num_items), 
                            squeeze=False)
    
    for idx in range(min(num_items, len(sentences))):
        words = sentences[idx].split()
        # Ensure we're working with a 2D array for heatmap
        attn_data = attention_np[idx, :len(words)]
        if len(attn_data.shape) > 1:
            # If attn_data is already 2D, use it directly
            sns.heatmap(attn_data, 
                        ax=axes[idx, 0],
                        cmap='viridis',
                        xticklabels=words,
                        yticklabels=words,  # Show words on both axes
                        cbar_kws={'label': 'Attention Weight'})
        else:
            # If attn_data is 1D, reshape to 2D for heatmap
            sns.heatmap(attn_data.reshape(1, -1), 
                        ax=axes[idx, 0],
                        cmap='viridis',
                        xticklabels=words,
                        yticklabels=['Attention'],
                        cbar_kws={'label': 'Attention Weight'})
        axes[idx, 0].set_title(f'Sentence {idx + 1}: {sentences[idx]}')
        
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_layer_attention(attention_weights: torch.Tensor,
                        sentences: List[str],
                        layers: List[int] = None,
                        save_path: str = None):
    """
    Plot attention patterns for specific layers.
    """
    if layers is None:
        layers = [0, 15, 31]  # First, middle, last layer
        
    attention_np = attention_weights.cpu().numpy()
    num_sentences = len(sentences)
    
    # Check dimensions
    if len(attention_np.shape) < 3:
        # If we only have a 2D matrix, add a dimension for layers
        attention_np = attention_np[None, :]
    
    fig, axes = plt.subplots(num_sentences, len(layers),
                            figsize=(5*len(layers), 4*num_sentences))
    
    if num_sentences == 1 and len(layers) == 1:
        axes = np.array([[axes]])
    elif num_sentences == 1:
        axes = axes[None, :]
    elif len(layers) == 1:
        axes = axes[:, None]
    
    for sent_idx, sent in enumerate(sentences):
        words = sent.split()
        for layer_idx, layer in enumerate(layers):
            if layer < attention_np.shape[0]:
                layer_data = attention_np[layer]
                # Ensure we have the right dimensions
                if len(layer_data.shape) >= 2 and sent_idx < layer_data.shape[0]:
                    attn = layer_data[sent_idx, :len(words)]
                else:
                    attn = layer_data[:len(words)]
                
                # Ensure we have a 2D matrix for heatmap
                if len(attn.shape) == 1:
                    attn = attn.reshape(1, -1)
                
                sns.heatmap(attn,
                           ax=axes[sent_idx, layer_idx],
                           cmap='viridis',
                           xticklabels=words,
                           yticklabels=['Attention'],
                           cbar_kws={'label': 'Weight'})
                axes[sent_idx, layer_idx].set_title(f'Layer {layer}')
            else:
                axes[sent_idx, layer_idx].text(0.5, 0.5, f"Layer {layer} not available",
                                              ha='center', va='center')
                axes[sent_idx, layer_idx].axis('off')
            
    plt.suptitle('Attention Patterns Across Layers', y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load model and prepare data
    model, tokenizer = load_llama()
    
    sentences = [
        "This is a tokenization example",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    # Get attention weights
    word_mappings, encodings = encode_input(model, tokenizer, prompt_task2, sentences)
    token_level_attention, _ = extract_token_attention(model, tokenizer, encodings)
    word_level_attention = extract_word_attention(word_mappings, token_level_attention)
    
    # Create output directory
    output_dir = Path("/scratch/7982399/thesis/outputs/attention_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot averaged attention
    averaged_attention = aggregate_attention(tokenizer, word_level_attention)
    plot_attention_heatmap(
        averaged_attention,
        sentences,
        save_path=str(output_dir / "averaged_attention.png"),
        title="Average Attention Across All Layers"
    )
    
    # Plot layer-wise attention
    layer_attention = aggregate_attention(tokenizer, word_level_attention)
    plot_layer_attention(
        layer_attention,
        sentences,
        layers=[0, 15, 31],
        save_path=str(output_dir / "layer_attention.png")
    )