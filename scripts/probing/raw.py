import torch
from load_model import *
from typing import List, Tuple, Dict
from prompts import prompt_task2, prompt_task3

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

def encode_input(model, tokenizer, prompt, sentences):
    """Encode all sentences including a prompt"""
    # Prepare input texts with prompt
    input_texts = [prompt + sentence for sentence in sentences]
    
    # Encode all sentences at once
    tokenized = tokenizer(
        input_texts,
        padding=True,
        return_tensors='pt',
        truncation=False
    )
    
    # Get word mappings
    word_mappings = map_tokens_to_words(tokenized.encodings)
    
    # Move to device
    encodings = {k: v.to(model.device) for k, v in tokenized.items()}
    
    return word_mappings, encodings

def extract_token_attention(model, tokenizer, encodings, decoding=True):

    with torch.no_grad():


        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
        generated_text = ["Decoding disabled."] * encodings['input_ids'].size(0)

        if decoding:
            # Generate one token for each sentence in batch
            next_tokens = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            # Decode next tokens while reusing past key values
            output = model(input_ids=next_tokens,
                         past_key_values=output.past_key_values,
                         output_attentions=True,
                         use_cache=True)
            
            # Use attention from the decoding step
            attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, 1, seq_len]
            generated_text = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
        
        return attention, generated_text
    
def extract_word_attention(
        word_mappings: List[List[List[int]]],
        generated_attention: torch.Tensor
) -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights.
    
    Args:
        word_mappings: List of token indices for each word in each sentence
        generated_attention: Attention tensor [num_layers, batch_size, num_heads, seq/1, seq_len]
    
    Returns:
        Word attention tensor [num_layers, batch_size, num_heads, query_len, max_words]
    """
    num_layers, batch_size, num_heads, query_len, _ = generated_attention.shape
    
    # Get maximum number of words across all sentences
    max_words = max(len(sentence_map) for sentence_map in word_mappings)

    # Initialize word-level attention tensor
    word_attentions = torch.zeros(
        (num_layers, batch_size, num_heads, query_len, max_words), 
        device=generated_attention.device
    )

    # Loop through each sentence in batch
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, token_group in enumerate(word_map):
            # Average token-level attentions for the word across all layers and heads
            word_attention = torch.zeros(
                (num_layers, num_heads, query_len), 
                device=generated_attention.device
            )
            
            for token_idx in token_group:
                # Get attention for this token
                token_attention = generated_attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention
            
            # Normalize by number of tokens in the word
            word_attention = word_attention / len(token_group)
            
            # Store averaged attention at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    return word_attentions 

def normalize_attention(
        attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize attention weights using:
    1. Z-score normalization across heads
    2. Square root of sentence lengths normalization
    
    Args:
        attention_weights: Attention tensor [num_layers, batch_size, num_heads, query_len, max_words]
        
    Returns:
        Normalized attention tensor [num_layers, batch_size, num_heads, query_len, max_words]
    """
    # Length normalization
    sentence_lengths = torch.count_nonzero(attention_weights, dim=-1)  # [num_layers, batch_size, num_heads, query_len]
    sqrt_length = torch.sqrt(sentence_lengths.clamp(min=1.0))  # avoid 0 div
    length_norm = attention_weights / sqrt_length.unsqueeze(-1)
    
    # Z-score normalization across heads
    head_mean = length_norm.mean(dim=2, keepdim=True)  # [num_layers, batch_size, 1, query_len, max_words]
    head_std = length_norm.std(dim=2, keepdim=True)  # [num_layers, batch_size, 1, query_len, max_words]
    head_std = head_std.clamp(min=1e-9) # avoid 0 div
    normalized_attention = (length_norm - head_mean) / head_std
    
    return normalized_attention

def aggregate_attention(
    attention_weights: torch.Tensor,
    average_layers: bool = True,
) -> torch.Tensor:
    # First normalize
    normalized_attention = normalize_attention(attention_weights)

    head_averaged = normalized_attention.mean(dim=2)  # [num_layers, batch_size, query_len, max_words]
    head_averaged = head_averaged.squeeze(2) # [num_layers, batch_size, max_words]
    
    s1 = head_averaged.shape
    head_averaged = exclude_prompt(head_averaged, prompt_task2)
    print(f"Excluding the prompt changes the shape from {s1} to {head_averaged.shape}")
    # Optional layer average
    if average_layers:
        return head_averaged.mean(dim=0)  # [batch_size, max_words]
    else:
        return head_averaged  # [num_layers, batch_size, max_words]
    #TODO: remove prompt attention before comparison

def exclude_prompt(
    attention_weights: torch.Tensor,
    prompt: str,
) -> torch.Tensor:
    """
    Exclude attention weights for the prompt tokens.
    
    Args:
        attention_weights: Attention tensor [num_layers, batch_size, num_heads, query_len, max_words]
        prompt_length: Number of tokens in the prompt
        
    Returns:
        Attention tensor without prompt tokens [num_layers, batch_size, num_heads, query_len, max_words]
    """

    prompt_tokens = tokenizer([prompt], padding=False, return_tensors='pt')
    prompt_mappings = map_tokens_to_words(prompt_tokens.encodings)
    prompt_length =  len(prompt_mappings[0])
    return attention_weights[..., prompt_length:]

if __name__ == "__main__":

    torch.cuda.empty_cache()
    # torch.set_default_device('cuda:0')
    model, tokenizer = load_llama()
    #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    sentences = [
        "This is a tokenization example",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    word_mappings, encodings = encode_input(model, tokenizer, prompt_task2, sentences)

    token_level_attention, generated_text = extract_token_attention(model, tokenizer, encodings)
    print(generated_text)
    print("Shape in token-level: ", token_level_attention.shape)
    
    word_level_attention = extract_word_attention(word_mappings, token_level_attention)
    print("Shape in word-level: ", word_level_attention.shape)

    # Get attention averaged across all layers
    averaged_attention = aggregate_attention(word_level_attention, average_layers=False)
    #print(f"Shape with averaged layers: {averaged_attention.shape}")  # [batch, seq_len]