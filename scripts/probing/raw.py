import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_model import load_llama
from typing import List, Tuple, Dict
from collections import defaultdict

def map_tokens_to_words(encoded_batch):
    mappings = []
    
    for encoded in encoded_batch:
        word_map = []
        for word_id in encoded.word_ids:
            if word_id is not None:
                start, end = encoded.word_to_tokens(word_id)
                if start == end - 1:
                    tokens = [start]
                else:
                    tokens = [start, end - 1]
                if len(word_map) == 0 or word_map[-1] != tokens:
                    word_map.append(tokens)
        mappings.append(word_map)
    
    return mappings

def get_sentence_attention(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        encoded_batch
) -> Tuple[List[List[str]], torch.Tensor]:
    
    word_mappings = map_tokens_to_words(encoded_batch.encodings)  # Extract token-to-word mapping before moving to device
    encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**encoded_batch, output_attentions=True, return_dict=True)
    
    attention_weights = torch.stack(outputs.attentions)  # Shape: [num_layers, batch, num_heads, seq_len, seq_len]
    num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Initialize word-level attention tensor (same shape as original) #TODO: we want word length not token length but worst case scenario we have trailing 0s
    word_attentions = torch.zeros((num_layers, batch_size, num_heads, seq_len, seq_len), device=model.device)

    # Loop through each sentence in batch
    for sentence_idx, word_map in enumerate(word_mappings):

        for word_idx, token_group in enumerate(word_map):  

            # Sum token-level attentions for the word across all layers and heads
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=model.device)
            
            for token_idx in token_group:
                token_attention = attention_weights[:, sentence_idx, :, :, token_idx]  # How much attention word j receives from all other words i?
                word_attention += token_attention  # Sum token attentions for the word

            # Store summed attention at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention #Keep all layers & heads

    # Convert token IDs back to tokens
    tokens_batch = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in encoded_batch["input_ids"].tolist()]
    print("Final shape: ", word_attentions.shape)
    return tokens_batch, word_attentions  # Shape: [num_layers, batch, num_heads, seq_len, seq_len]

def aggregate_attention(
    attention_weights: torch.Tensor,
    mode = "sum"
) -> torch.Tensor:
    """
    Aggregate attention weights across heads and words.
    
    Args:
        attention_weights: Attention weights from model
            [layer x batch x heads x seq_len x seq_len]
    
    Returns:
        Aggregated attention weights [layers x batch x seq_len]
    """
    num_layers = len(attention_weights)
    aggregated_attention = []
    
    for layer in range(num_layers):
        # Get attention weights for current layer [batch, heads, seq_len, seq_len]
        layer_attention = attention_weights[layer]
        
        # Average across attention heads [batch, seq_len, seq_len]
        head_average = layer_attention.mean(dim=1)
        
        # Average (or sum) across words [batch, seq_len]
        word_average = head_average.sum(dim=1) if mode == "sum" else head_average.mean(dim=1)
        
        aggregated_attention.append(word_average)
    
    # Stack layers [layers, batch, seq_len]
    return torch.stack(aggregated_attention, dim=0)

if __name__ == "__main__":

    torch.cuda.empty_cache()
    # torch.set_default_device('cuda:0')
    model, tokenizer = load_llama()

    sentences = [
        "This is a tokenization example",
        "The quick brown fox jumps over the lazy dog.",
    ]
    #TODO: look into alternatives for truncation
    encoded_batch = tokenizer(sentences, padding=True, return_tensors='pt', truncation=False) 
 
    # Get attention patterns
    tokens, attention = get_sentence_attention(model, tokenizer, encoded_batch)
    
    # Aggregate attention
    attention_agg = aggregate_attention(attention)
    
    # Print results
    print("\nAttention shape analysis:")
    print(f"Extracted attention: {attention.shape}")
    print(f"Final aggregated shape: {attention_agg.shape}")