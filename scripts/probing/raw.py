import torch
from scripts.probing.load_model import *
from typing import List, Tuple
from materials.prompts import prompt_task2, prompt_task3
import re
import json

def encode_input(sentences: List[List[str]], tokenizer: AutoTokenizer, task: str):
    """
    Encodes input sentences for model processing using batch tokenization.

    Args:
        sentences: List of pretokenized sentences (List[List[str]])
        tokenizer: The tokenizer to use
        task: The task type ('none', 'task2' or 'task3')

    Returns:
        Tuple of (batch_encodings, number of prompt tokens)
    """
    if task == "none":
        prompt_words = []
        sentences_to_encode = [[words[0]] + [" " + word for word in words[1:]] for words in sentences]

    elif task == "task2":
        prompt_words = re.sub(r'[^\w\s]', '', prompt_task2).split()
        sentences_to_encode = [prompt_words + sentence for sentence in sentences]
        sentences_to_encode = [[words[0]] + [" " + word for word in words[1:]] for words in sentences_to_encode]

    elif task == "task3":
        sentences_to_encode = []
        for item in sentences:
            sent, relation = item["sentence"], item["relation_type"]
            prompt_words = re.sub(r'[^\w\s]', '', prompt_task3.format(relation)).split()
            combined = prompt_words + sent
            sentences_to_encode.append([combined[0]] + [" " + word for word in combined[1:]])
    
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'none', 'task2' or 'task3'.")

    # Perform batch tokenization with pretokenized input
    batch_encodings = tokenizer(
        sentences_to_encode,
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
        is_split_into_words=True  # pretokenized input
    )

    # Return the encodings and the number of prompt tokens
    return batch_encodings, sentences_to_encode, len(prompt_words)

def get_word_mappings(sentences: List[List[str]], batch_encodings, tokenizer: AutoTokenizer) -> List[List[Tuple[str, int]]]:
    """
    Calculates word mappings for each sentence, mapping each word to the number of tokens assigned to it.

    Args:
        sentences: List of original sentences as lists of strings (List[List[str]])
        batch_encodings: The batch encodings from the tokenizer
        tokenizer: The tokenizer used for encoding

    Returns:
        List of word mappings for each sentence. Each mapping is a list of tuples (original_word, num_tokens_assigned).
    """
    word_mappings = []
    input_ids = batch_encodings["input_ids"].tolist()  # Convert tensor to list for processing

    for sentence_idx, sentence in enumerate(sentences):
        token_sequence = input_ids[sentence_idx]  # Token sequence for the current sentence
        token_idx = 0
        sentence_mapping = []

        for word in sentence:
            num_tokens = 0
            tokens = []

            while token_idx < len(token_sequence):
                token = tokenizer.convert_ids_to_tokens(token_sequence[token_idx])
                if num_tokens > 0 and (token.startswith("Ġ") or token.startswith("Ċ")):
                    # A new word starts, break the loop
                    break
                tokens.append(token)
                num_tokens += 1
                token_idx += 1
            sentence_mapping.append((word, num_tokens, tokens))
        word_mappings.append(sentence_mapping)
    
    return word_mappings
            

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
        for word_idx, (word, num_tokens, tokens) in enumerate(word_map):
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
    model_type = "causal"

    model, tokenizer = load_llama(model_type=model_type)
    #save_model(model, tokenizer, f"/scratch/7982399/hf_cache/{task}")

    # Load the sentences
    with open(f'materials/sentences_{task}.json', 'r') as f:
        sentences = json.load(f)
    
    # Subset for testing
    if subset:
        sentences = sentences[:subset]

    encodings, sentences_full, prompt_len = encode_input(sentences, tokenizer, task)
    word_mappings = get_word_mappings(sentences_full, encodings, tokenizer)
    attention = get_attention(model, encodings)

    torch.save({
        'attention': attention,
        'input_ids': encodings['input_ids'],
        'word_mappings': word_mappings,
        'prompt_len': prompt_len
    }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

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