import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3
import re
import json

def encode_input(sentences: List[List[str]], tokenizer: AutoTokenizer, task: str, relation_type: str = None):
    """
    Encodes pretokenized input for model processing.
    
    Args:
        sentences: List of pretokenized sentences (List[List[str]])
        tokenizer: The tokenizer to use
        task: The task type ('none', 'task2' or 'task3')
        relation_type: Relation type for task3 (required if task is 'task3')
        
    Returns:
        Tuple of (batch_encodings, word_mappings, number of prompt tokens)
    """
    if task == "none":
        prompt_words = []

    elif task == "task2":
        prompt_words = re.sub(r'[^\w\s]', '', prompt_task2).split()
        sentences_to_encode = [prompt_words + sentence for sentence in sentences]

    elif task == "task3":
        sentences_to_encode = []
        for item in sentences:
            sent, relation = item["sentence"], item["relation_type"]
            prompt_words = re.sub(r'[^\w\s]', '', prompt_task3.format(relation)).split()
            sentences_to_encode.append(prompt_words + sent)
    
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'none', 'task2' or 'task3'.")

    batch_encodings = []
    word_mappings = []
    
    for sentence in sentences_to_encode:
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

def process_attention(attention: torch.Tensor, word_mappings: List[List[Tuple[str, int]]], prompt_len: int, reduction: str = "mean") -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights and reduce over heads.

    Args:
        attention: Attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
        word_mappings: List of token counts for each word in each sentence
        prompt_len: Length of prompt, to know the number of tokens in the beginning to filter out
        reduction: Reduction method, either "mean" (average) or "max" (max of tokens per word)
    Returns:
        Word attention tensor [num_layers, batch_size, max_words]
    """
    num_layers, batch_size, num_heads, seq_len, _ = attention.shape
    device = attention.device

    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = torch.zeros((num_layers, batch_size, num_heads, seq_len, max_words), device=device)
    
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens) in enumerate(word_map):
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=device)
            token_attentions = []
            
            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                token_idx = sum(token[1] for token in prev_tokens) + n_token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                token_attentions.append(token_attention)
            
            token_attentions = torch.stack(token_attentions, dim=-1)  # Shape: [num_layers, num_heads, seq_len, num_tokens]
            
            if reduction == "mean":
                word_attention = token_attentions.mean(dim=-1)
            elif reduction == "max":
                word_attention = token_attentions.max(dim=-1).values
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")
            
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention
    
    word_average = word_attentions.mean(dim=3)  # [num_layers, batch_size, num_heads, seq_len, max_words]
    head_average = word_average.mean(dim=2)  # [num_layers, batch_size, seq_len, max_words]
    head_average = head_average.squeeze(2)  # [num_layers, batch_size, max_words]
    
    attention_sum = head_average.sum(dim=2, keepdim=True)  # Sum over words
    normalized_attention = head_average / (attention_sum + 1e-8)  # Normalize

    return normalized_attention[:, :, prompt_len:]

subset = False # Set to False to process all sentences
if __name__ == "__main__":

    task = "task3" # None, task2, task3
    # model_type = "causal"

    # model, tokenizer = load_llama(model_type=model_type)
    # # #save_model(model, tokenizer, f"/scratch/7982399/hf_cache/{task}")

    # # # Load the sentences
    # with open(f'materials/sentences_{task}.json', 'r') as f:
    #     sentences = json.load(f)
    
    # # Subset for testing
    # if subset:
    #     sentences = sentences[:subset]

    # encodings, word_mappings, prompt_len = encode_input(sentences, tokenizer, task)

    # attention = get_attention(model, encodings)

    # torch.save({
    #     'attention': attention,
    #     'input_ids': encodings['input_ids'],
    #     'word_mappings': word_mappings,
    #     'prompt_len': prompt_len
    # }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

    # Load the saved dictionary
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

    # Extract each component
    attention = loaded_data['attention']
    input_ids = loaded_data['input_ids']
    word_mappings = loaded_data['word_mappings']
    prompt_len = loaded_data['prompt_len']

    attention_processed = process_attention(attention, word_mappings, prompt_len, reduction="max")
    print("Shape: ", attention_processed.shape)
    
    torch.save({
        'attention_processed': attention_processed,
        'input_ids': input_ids,
        'word_mappings': word_mappings,
        'prompt_len': prompt_len
    }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_processed.pt")