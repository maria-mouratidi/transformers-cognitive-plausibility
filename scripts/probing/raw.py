import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3

def encode_input(sentence, prompt, tokenizer, special_tokens=True):
    
    if prompt:
        sentence = prompt + sentence
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=special_tokens)
    individual_tokens = tokenizer.batch_encode_plus(sentence.split(), add_special_tokens=False)['input_ids']

    tokens_per_word = list(zip(sentence.split(), [len(item) for item in individual_tokens]))
    tokens = inputs['input_ids']

    token_indices = []
    current_index = 0
    token_groups = []
    
    for word, token_count in tokens_per_word:
        indices = list(range(current_index, current_index + token_count))
        token_indices.append((word, indices))
        token_groups.append(indices)
        current_index += token_count

    return tokens, token_indices, token_groups

def get_attention(model, encodings, decoding=True):

    with torch.no_grad():

        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
        generated_text = ["Decoding disabled."] * encodings['input_ids'].size(0)

        if decoding:
            # Generate one token for each sentence in batch #TODO: Change to generate multiple tokens
            next_tokens = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            # Decode next tokens while reusing past key values
            output = model(input_ids=next_tokens,
                         past_key_values=output.past_key_values,
                         output_attentions=True,
                         use_cache=True)
            
            # Use attention from the decoding step
            attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, 1, seq_len]
        
        return attention, next_tokens
    
def wordlevel_attention(word_mappings: List[List[List[int]]], attention: torch.Tensor
) -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights.
    
    Args:
        word_mappings: List of token indices for each word in each sentence
        generated_attention: Attention tensor [num_layers, batch_size, num_heads, seq/1, seq_len]
    
    Returns:
        Word attention tensor [num_layers, batch_size, num_heads, seq_len, max_words]
    """

    layers, batches, heads, seq_len, _ = attention.shape
    device = attention.device
    
    # Get maximum number of words across all sentences
    max_words = max(len(word_map) for word_map in word_mappings)

    # Initialize word-level attention tensor
    word_attentions = torch.zeros((layers, batches, heads, seq_len, max_words), device=device)

    # Loop through each sentence in batch
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, token_group in enumerate(word_map):
            # Average token-level attentions for the word across all layers and heads
            word_attention = torch.zeros((layers, heads, seq_len), device=attention.device)
            
            for token_idx in token_group:
                # Get attention for this token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention
            
            # Normalize by number of tokens in the word
            word_attention = word_attention / len(token_group)
            
            # Store averaged attention at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    return word_attentions 

def process_attention(
    tokenizer: transformers.PreTrainedTokenizer,
    attention_weights: torch.Tensor,
    average_layers: bool = False
) -> torch.Tensor:
    
    # First normalize
    normalized_attention = normalize_attention(attention_weights)

    # Average across heads
    head_averaged = normalized_attention.mean(dim=2)  # [num_layers, batch_size, seq_len, max_length]
    head_averaged = head_averaged.squeeze(2) # [num_layers, batch_size, max_words]
    
    s1 = head_averaged.shape
    head_averaged = exclude_prompt(tokenizer, head_averaged, prompt_task2)
    print(f"Excluding the prompt changes the shape from {s1} to {head_averaged.shape}")
    
    # Optional layer average
    if average_layers:
        return head_averaged.mean(dim=0)  # [batch_size, max_length]
    else:
        return head_averaged  # [num_layers, batch_size, max_length]

def exclude_prompt(tokenizer, attention_weights: torch.Tensor, prompt: str) -> torch.Tensor:
    """
    Exclude attention weights for the prompt tokens.
    """

    prompt_tokens = tokenizer([prompt], padding=False, return_tensors='pt')
    prompt_mappings = map_tokens_to_words(prompt_tokens.encodings)
    prompt_length =  len(prompt_mappings[0])
    return attention_weights[..., prompt_length:]

if __name__ == "__main__":

    model, tokenizer = load_llama()
    #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    sentences = [
        "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
        "She won a Novel Prize in 1911."
    ]

    for sentence in sentences:
        print(encode_input(sentence, prompt_task2, tokenizer))
    
    
    # word_mappings, word_token_dict, encodings = encode_input(model, tokenizer, prompt_task2, sentences)
    # print("mapping dict: " , word_token_dict)

    # token_level_attention, generated_text = extract_token_attention(model, tokenizer, encodings)
    
    # word_level_attention = extract_word_attention(word_mappings, token_level_attention)
    # print("Shape: ", word_level_attention.shape)

    # # Get attention averaged across all layers
    # averaged_attention = aggregate_attention(tokenizer, word_level_attention)
    # torch.save(averaged_attention, '/scratch/7982399/thesis/outputs/attention_tensor.pt')
    # print(f"Shape with averaged layers: {averaged_attention.shape}")  # [batch, seq_len]