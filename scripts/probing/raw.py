import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3
import re

def encode_input(sentences: List[str], tokenizer: AutoTokenizer, task: str, relation_type: str = None):

    if task == "task2":
        sentences = [f"Instruction: {prompt_task2} Context: {sentence}" for sentence in sentences]

    elif task == "task3":
        if relation_type is None:
            raise ValueError("Relation type must be provided for task3.")
        sentences = [f"Question: {prompt_task3} {relation_type} Context: {sentence}" for sentence in sentences]

    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'task2' or 'task3'.")

    sentence_tokens_per_word = [], []
    # Pretokenize input to align with human data
    sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences] # Remove punctuation
    sentences = [sentence.split() for sentence in sentences] # Split by whitespace

    for sentence in sentences:
        # Encode using pretokenized input
        individual_tokens = tokenizer.batch_encode_plus(sentence, add_special_tokens=False)['input_ids']
        # Count how many tokens correspond to each word
        tokens_per_word = list(len(item) for item in individual_tokens)
        sentence_tokens_per_word.append(tokens_per_word)

    batch_encodings = tokenizer(sentences, return_tensors='pt', is_split_into_words=True, padding=True, truncation=False, add_special_tokens=False) 
 
    return batch_encodings, sentences, sentence_tokens_per_word


def get_prompt_token_count(tokenizer, task, relation_type=None):
    """
    Count the number of tokens in the prompt prefix for a given task.
    
    Args:
        tokenizer: The tokenizer to use
        task: The task ("task2" or "task3")
        relation_type: For task3, the relation type string
        
    Returns:
        Number of tokens in the prompt
    """
    if task == "task2":
        prompt_text = f"Instruction: {prompt_task2} Context: "
    elif task == "task3":
        prompt_text = f"Question: {prompt_task3} {relation_type} Context: "
    else:
        raise ValueError(f"Invalid task: {task}")
    
    prompt_text = re.sub(r'[^\w\s]', '', prompt_text) # Remove punctuation
    prompt_text = prompt_text.split() # Split by whitespace

    # Get tokens for just the prompt
    tokens = tokenizer(prompt_text, add_special_tokens=False, is_split_into_words=True)['input_ids']
    return len(tokens)

def get_attention(model, encodings):

    with torch.no_grad():

        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
       
    return attention
    
def process_attention(attention: torch.Tensor, word_mappings: List[List[List[int]]]) -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights and average over heads.

    Args:
        attention: Attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
        word_mappings: List of token counts for each word in each sentence
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
        token_pos = 0

        for word_idx, num_tokens in enumerate(word_map):
            
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=attention.device)
            
            for n_token in range(num_tokens):
                token_idx = token_pos + n_token
                # Get attention for this token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention #TODO: CHECK if this really sums the word-level values
            token_pos += num_tokens
            
            # Average from all tokens
            word_attention = word_attention / num_tokens
            
            # Store at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    word_average = word_attentions.mean(dim=3)  # [num_layers, batch_size, num_heads, max_words]
    head_average = word_average.mean(dim=2)  # [num_layers, batch_size, seq_len, max_words]
    head_average = head_average.squeeze(2) # [num_layers, batch_size, max_words]

    return head_average


if __name__ == "__main__":

    task = "task2"
    model, tokenizer = load_llama(task)
    # #model_task3, tokenizer = load_llama(model_type = "qa")


    # #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    # sentences = [
    #     "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
    #     "She won a Novel Prize in 1911."
    # ]

    # encodings, wordlists, token_word_mappings = encode_input(sentences, tokenizer, "task2")

    # attention = get_attention(model_task2, encodings)

    # torch.save({
    #     'attention': attention,
    #     'sentence_tokens_per_word': token_word_mappings,
    #     'sentence_words': wordlists
    # }, "/scratch/7982399/thesis/outputs/attention_data.pt")


    # Load the saved dictionary
    loaded_data = torch.load("/scratch/7982399/thesis/outputs/attention_data.pt")

    # Extract each component
    attention = loaded_data['attention']
    word_mappings = loaded_data['sentence_tokens_per_word']
    sentence_words = loaded_data['sentence_words']

    print("Attention tensor before preprocessing: ", attention.shape)

    prompt_boundary = get_prompt_token_count(tokenizer, task)
    attention_processed = process_attention(attention, word_mappings)
    
    print("Attention tensor after preprocessing: ", attention_processed.shape)

    # torch.save(attention, "/scratch/7982399/thesis/outputs/attention.pt")

    #TODO: define relation types