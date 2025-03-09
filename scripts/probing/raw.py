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

        # Decode individual tokens for comparison
        decoded_individual_tokens = [tokenizer.decode(tokens) for tokens in individual_tokens]
        print(f"Decoded Individual Tokens: {decoded_individual_tokens}\n\n")

    # Encode all sentences together
    batch_encodings = tokenizer(sentences, 
                              return_tensors='pt', 
                              padding=True,
                              truncation=False,
                              add_special_tokens=True)
    
    # Decode batch tokens for comparison
    batch_tokens = batch_encodings['input_ids']
    decoded_batch_tokens = [[tokenizer.decode([token]) for token in tokens] for tokens in batch_tokens]
    print(f"Decoded Batch Tokens: {decoded_batch_tokens}\n\n")


 
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
        attention: Attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
    
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
            # Average token-level attentions for the word across all layers and heads
            word_attention = torch.zeros((num_layers, num_heads, seq_len), device=attention.device)
            
            for n_token in range(num_tokens):
                token_idx = token_pos + n_token
                continue
                # Get attention for this token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                word_attention += token_attention #TODO: CHECK if this really sums the word-level values
            token_pos += num_tokens
            
            # Average from all tokens
            word_attention = word_attention / num_tokens
            
            # Store at the word position
            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    word_average = word_attentions.mean(dim=4)  # [num_layers, batch_size, num_heads, seq_len]

    head_average = word_average.mean(dim=2)  # [num_layers, batch_size, seq_len, max_length]
    head_average = head_average.squeeze(2) # [num_layers, batch_size, max_words]

    return head_average

if __name__ == "__main__":

    model_task2, tokenizer = load_llama(model_type = "causal")
    # #model_task3, tokenizer = load_llama(model_type = "qa")

    # #save_model(model, tokenizer, "/scratch/7982399/hf_cache")

    sentences = [
        "As a child, his hero was Batman, and as a teenager his interests shifted towards music.",
        "She won a Novel Prize in 1911."
    ]


    encodings, wordlists, token_word_mappings = encode_input(sentences, tokenizer, "task2")

    # attention = get_attention(model_task2, encodings)

    # torch.save({
    #     'attention': attention,
    #     'sentence_tokens_per_word': token_word_mappings,
    #     'sentence_words': wordlists
    # }, "/scratch/7982399/thesis/outputs/attention_data.pt")


    # # Load the saved dictionary
    # loaded_data = torch.load("/scratch/7982399/thesis/outputs/attention_data.pt")

    # # Extract each component
    # attention = loaded_data['attention']
    # word_mappings = loaded_data['sentence_tokens_per_word']
    # sentence_words = loaded_data['sentence_words']

    # print(attention.shape)
    # print(word_mappings)
    # attention_processed = process_attention(word_mappings, attention)

    # # Exclude prompt tokens
    # prompt_len = len(prompt.split())
    # attention = attention[..., prompt_len:]

    # torch.save(attention, "/scratch/7982399/thesis/outputs/attention.pt")

    # print(attention_processed.shape)

    #TODO: be able to load attention tensor
    #TODO: define relation types
    #TODO: update averaging for square matrix