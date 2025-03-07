import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3

def encode_input(sentences, prompt, tokenizer, special_tokens=True):
    if isinstance(sentences, str):
        sentences = [sentences]  # Convert single sentence to list
    
    # Create full texts with prompt
    full_texts = [prompt + sentence for sentence in sentences]
    
    # Batch encode with padding
    inputs = tokenizer.batch_encode_plus(
        full_texts,
        return_tensors='pt',
        padding=True,
        add_special_tokens=special_tokens
    )
    
    all_token_indices = []
    
    # For each sentence
    for i, sentence in enumerate(sentences):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_ids)
        
        # Get the mapping for words to token positions using the actual tokenization
        token_indices = []
        words = sentence.split()
        
        current_pos = prompt_len
        for word in words:
            # Get tokens for this word
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            word_len = len(word_ids)
            indices = list(range(current_pos, current_pos + word_len))
            token_indices.append((word, indices))
            
            current_pos += word_len

        all_token_indices.append(token_indices)
    
    return inputs, all_token_indices

def get_attention(model, encodings, decoding=True):

    with torch.no_grad():

        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]

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
    
def process_attention(word_mappings: List[List[List[int]]], attention: torch.Tensor) -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights and average over heads.

    Args:
        word_mappings: List of token indices for each word in each sentence
        attention: Attention tensor [num_layers, batch_size, num_heads, seq_len/1, seq_len]
    
    Returns:
        Word attention tensor [num_layers, batch_size, num_heads, seq_len, max_length]
    """

    num_layers, batch_size, num_heads, seq_len, _ = attention.shape
    device = attention.device
    
    # Get maximum number of words across all sentences
    max_words = max(len(word_map) for word_map in word_mappings)

    # Initialize word-level attention tensor
    word_attentions = torch.zeros((num_layers, batch_size, num_heads, seq_len, max_words), device=device)

    # Loop through each sentence in batch
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, token_group) in enumerate(word_map):
            # Average token-level attentions for the word across all layers and heads
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=attention.device)
            
            for token_idx in token_group:
                # Get attention for this token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention
            
            # Average from all tokens
            word_attention = word_attention / len(token_group)
            
            # Store at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    head_average = word_attentions.mean(dim=2)  # [num_layers, batch_size, seq_len, max_length]
    head_average = head_average.squeeze(2) # [num_layers, batch_size, max_words]

    return head_average

if __name__ == "__main__":

    model, tokenizer = load_llama()
    prompt = prompt_task2
    #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    sentences = [
        "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
        "She won a Novel Prize in 1911."
    ]

    encodings, word_mappings, token_groups = encode_input(sentences, prompt, tokenizer)

    print(word_mappings, "\n", token_groups)

    attention, _ = get_attention(model, encodings)

    attention_processed = process_attention(word_mappings, attention)

    # # Exclude prompt tokens
    # prompt_len = len(prompt.split())
    # attention = attention[..., prompt_len:]

    torch.save(attention, "/scratch/7982399/thesis/outputs/attention.pt")

    print(attention_processed.shape)