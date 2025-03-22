import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3
import re
import json

subset = 5
print("Using a subset of the data for testing: ", subset)

def encode_input(sentences: List[List[str]], tokenizer: AutoTokenizer, task: str, relation_type: str = None):
    """
    Encodes pretokenized input for model processing.
    
    Args:
        sentences: List of pretokenized sentences (List[List[str]])
        tokenizer: The tokenizer to use
        task: The task type ('task2' or 'task3')
        relation_type: Relation type for task3 (required if task is 'task3')
        
    Returns:
        Tuple of (batch_encodings, word_mappings, number of prompt tokens)
    """
    if task == "task2":
        prompt_words = re.sub(r'[^\w\s]', '', prompt_task2).split()
        sentences = [prompt_words + sentence for sentence in sentences]

    elif task == "task3":
        if relation_type is None:
            raise ValueError("Relation type must be provided for task3.")
        prompt_words = re.sub(r'[^\w\s]', '', prompt_task3 + relation_type).split()
        sentences = [prompt_words + sentence for sentence in sentences]
    
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'task2' or 'task3'.")

    batch_encodings = []
    word_mappings = []
    
    for sentence in sentences:
        # Track word to token mapping
        word_to_tokens = []
        all_tokens = []
        
        # Process each word individually
        for word in sentence:
            # Get tokens for this word
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            # Store the mapping (word, token_count)
            word_to_tokens.append((word, len(word_tokens)))
            # Add these tokens to our running list
            all_tokens.extend(word_tokens)
        
        # Store the mapping for this sentence
        word_mappings.append(word_to_tokens)
        # Store the complete token sequence for this sentence
        batch_encodings.append(all_tokens)
    
    # Pad sequences to the same length
    max_length = max(len(tokens) for tokens in batch_encodings)
    padded_encodings = [tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)) 
                        for tokens in batch_encodings]
    
    # Convert to tensors
    input_ids = torch.tensor(padded_encodings)
    attention_mask = torch.tensor([[1] * len(tokens) + [0] * (max_length - len(tokens)) 
                                   for tokens in batch_encodings])
    
    # Create the encodings dictionary
    encodings = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    return encodings, word_mappings, len(prompt_words)


def get_attention(model, encodings):

    device = next(model.parameters()).device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    if hasattr(torch, 'compile') and device == 'cuda':
        model = torch.compile(model)

    with torch.inference_mode():

        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
       
    return attention
    
def process_attention(attention: torch.Tensor, word_mappings: List[List[Tuple[str, int]]], prompt_len) -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights and average over heads.

    Args:
        attention: Attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
        word_mappings: List of token counts for each word in each sentence
        prompt_len: Length of prompt, to know the number of tokens in the beginning to filter out
    Returns:
        Word attention tensor [num_layers, batch_size, max_words]
    """
    num_layers, batch_size, num_heads, seq_len, _ = attention.shape
    device = attention.device

    # Get maximum number of words across all sentences
    max_words = max(len(word_map) for word_map in word_mappings)
    # Initialize word-level attention tensor
    word_attentions = torch.zeros((num_layers, batch_size, num_heads, seq_len, max_words), device=device)
    
    # Loop through each sentence in batch
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens) in enumerate(word_map):
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=attention.device)
            
            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                # Count tokens before this word to get the token index
                token_idx = sum(token[1] for token in prev_tokens) + n_token
                # Get attention for this token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention
            # Average from all tokens
            word_attention = word_attention / num_tokens
            # Store at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    word_average = word_attentions.mean(dim=3)  # [num_layers, batch_size, num_heads, max_words]
    head_average = word_average.mean(dim=2)  # [num_layers, batch_size, seq_len, max_words]
    head_average = head_average.squeeze(2) # [num_layers, batch_size, max_words]
    
     # Normalize attention such that word attentions sum up to 1 in each sentence
    attention_sum = head_average.sum(dim=2, keepdim=True)  # Sum over words
    normalized_attention = head_average / (attention_sum + 1e-8)  # Add epsilon to avoid division by zero

    return normalized_attention[:, :, prompt_len:]  # Remove prompt words


if __name__ == "__main__":

    model_task2, tokenizer = load_llama(model_type="causal")
    # # model_task3, tokenizer = load_llama(model_type = "qa")
    # save_model(model_task2, tokenizer, "/scratch/7982399/hf_cache")

    # # Load the sentences
    # with open('materials/sentences.json', 'r') as f:
    #     sentences = json.load(f)
    # # Subset for testing
    # sentences = sentences[:subset]

    # encodings, word_mappings, prompt_len = encode_input(sentences, tokenizer, "task2")

    # attention = get_attention(model_task2, encodings)

    # torch.save({
    #     'attention': attention,
    #     'word_mappings': word_mappings,
    #     'prompt_len': prompt_len
    # }, "/scratch/7982399/thesis/outputs/attention_data.pt")

    # Load the saved dictionary
    loaded_data = torch.load("/scratch/7982399/thesis/outputs/attention_data.pt")

    # Extract each component
    attention = loaded_data['attention']
    word_mappings = loaded_data['word_mappings']
    prompt_len = loaded_data['prompt_len']


    attention_processed = process_attention(attention, word_mappings, prompt_len)
    print("Shape: ", attention_processed.shape)
    
    torch.save({
        'attention_processed': attention_processed,
        'word_mappings': word_mappings,
        'prompt_len': prompt_len
    }, "/scratch/7982399/thesis/outputs/attention_processed.pt")

    #TODO: define relation types
    #TODO: take max instead of average attention from subwords