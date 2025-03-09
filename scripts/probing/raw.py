import torch
from scripts.probing.load_model import *
from typing import List, Tuple, Dict
from materials.prompts import prompt_task2, prompt_task3

def encode_input(sentences: List[str], tokenizer: AutoTokenizer, task: str, relation_type: str = None):

    if task == "task2":
        sentences = [f"Instruction: {prompt_task2} Context: {sentence}" for sentence in sentences]

    elif task == "task3":
        if relation_type is None:
            raise ValueError("Relation type must be provided for task3.")
        sentences = [f"Question: {prompt_task3} {relation_type} Context: {sentence} Answer:" for sentence in sentences]

    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'task2' or 'task3'.")
     
    sentence_words, sentence_tokens_per_word = [], []
    
    for sentence in sentences: 

        wordlist = sentence.split()

        # Encode each individual word
        individual_tokens = tokenizer.batch_encode_plus(wordlist, add_special_tokens=False)['input_ids']
        # Count how many tokens correspond to each word
        tokens_per_word = list(len(item) for item in individual_tokens)
        sentence_words.append(wordlist)
        sentence_tokens_per_word.append(tokens_per_word)

    # Encode all sentences together
    batch_encodings = tokenizer(sentences, 
                              return_tensors='pt', 
                              padding=True,
                              truncation=True,
                              add_special_tokens=True)

 
    return batch_encodings, sentence_words, sentence_tokens_per_word

def get_attention(model, encodings):

    with torch.no_grad():

        # First forward pass with input prompt
        output = model(**encodings, output_attentions=True, use_cache=True)
        
        # Get attention from first pass
        attention = torch.stack(output.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
       
    return attention
    
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

    model_task2, tokenizer = load_llama(model_type = "causal")
    #model_task3, tokenizer = load_llama(model_type = "qa")

    #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    sentences = [
        "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
        "She won a Novel Prize in 1911."
    ]


    encodings, wordlists, token_word_mappings = encode_input(sentences, tokenizer, "task2")

    attention = get_attention(model_task2, encodings)

    torch.save(attention, "/scratch/7982399/thesis/outputs/attention_unprocessed.pt")

    print(attention.shape)

    # attention_processed = process_attention(word_mappings, attention)

    # # Exclude prompt tokens
    # prompt_len = len(prompt.split())
    # attention = attention[..., prompt_len:]

    # torch.save(attention, "/scratch/7982399/thesis/outputs/attention.pt")

    # print(attention_processed.shape)

    #TODO: be able to load attention tensor
    #TODO: define relation types
    #TODO: update averaging for square matrix