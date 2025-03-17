import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3
import re

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

    word_mappings = []

    for sentence in sentences:
        # Encode using pretokenized input
        individual_tokens = tokenizer.batch_encode_plus(sentence, add_special_tokens=False)['input_ids']
        # Count how many tokens correspond to each word
        mapping = [(word, len(item)) for word, item in zip(sentence, individual_tokens)]
        word_mappings.append(mapping)

    batch_encodings = tokenizer(sentences, return_tensors='pt', is_split_into_words=True, padding=True, truncation=False, add_special_tokens=False) 
 
    return batch_encodings, word_mappings, len(prompt_words)


def get_attention(model, encodings):

    with torch.no_grad():

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

    return head_average[:, :, prompt_len:]  # Remove prompt words


if __name__ == "__main__":

    # model_task2, tokenizer = load_llama(model_type="causal")
    # # model_task3, tokenizer = load_llama(model_type = "qa")


    # save_model(model_task2, tokenizer, "/scratch/7982399/hf_cache")

    # sentences = [
    #     "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
    #     "She won a Novel Prize in 1911."
    # ]

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

    print("Attention tensor before preprocessing: ", attention.shape)


    attention_processed = process_attention(attention, word_mappings, prompt_len)
    
    print("Attention tensor after preprocessing: ", attention_processed.shape)
    print(sentences[0][prompt_len:])
    print(attention_processed[0, 0, :])
    print(len(sentences[0][prompt_len:]), attention_processed[0, 0, :].shape)

    #torch.save(attention_processed, "/scratch/7982399/thesis/outputs/attention_processed.pt")

    #TODO: define relation types