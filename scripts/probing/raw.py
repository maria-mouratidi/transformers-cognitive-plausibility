import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_model import load_llama
from typing import List, Tuple, Dict
from collections import defaultdict

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

def get_sentence_attention(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        encoded_batch
) -> Tuple[List[List[str]], torch.Tensor]:
    
    word_mappings = map_tokens_to_words(encoded_batch.encodings) #before moving to device
    encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**encoded_batch, output_attentions=True, return_dict=True)
    
    attention_weights = outputs.attentions 
    word_attentions = []

    #TODO: maybe this can be done more efficiently.
    for sentence_idx, word_map in enumerate(word_mappings):
        print("Processing map: ", word_map)
        for word_idx, token_group in enumerate(word_map):  #index is the word idx, the item is a list of token(s)
            word_attention = torch.zeros()  #TODO: find the correct shape here
            print(f"Processing word: {word_idx} which corresponds to tokens {token_group}")
            for token_idx in token_group:
                token_attention = attention_weights[-1][sentence_idx][token_idx] #TODO: verify indexing represent each token attn
                print("This token has attention with shape: ", token_attention.shape)
                word_attention += token_attention #TODO: find the correct way to sum the token attentions
            word_attentions.append(word_attention)
    
    # Convert to tokens
    tokens_batch = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs["input_ids"].tolist()]
    
    return tokens_batch, attention_weights


def aggregate_attention(
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """
    Aggregate attention weights across heads and words.
    
    Args:
        attention_weights: Attention weights from model
            [layer x batch x heads x seq_len x seq_len]
    
    Returns:
        Aggregated attention weights [layers x batch x seq_len]
    """
    num_layers = len(attention_weights)
    aggregated_attention = []
    
    for layer in range(num_layers):
        # Get attention weights for current layer [batch, heads, seq_len, seq_len]
        layer_attention = attention_weights[layer]
        
        # Average across attention heads [batch, seq_len, seq_len]
        head_average = layer_attention.mean(dim=1)
        
        # Average across words [batch, seq_len]
        word_average = head_average.mean(dim=1)
        
        aggregated_attention.append(word_average)
    
    # Stack layers [layers, batch, seq_len]
    return torch.stack(aggregated_attention, dim=0)

if __name__ == "__main__":

    torch.cuda.empty_cache()
    # torch.set_default_device('cuda:0')
    model, tokenizer = load_llama()

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        #"She read the book that he recommended to them."
    ]
    
    encoded_batch = tokenizer(sentences, padding=True, return_tensors='pt', truncation=True) 
 
    # Get attention patterns
    print(get_sentence_attention(model, tokenizer, encoded_batch))
    
    # # Aggregate attention
    # result = aggregate_attention(attention)
    
    # # Print results
    # print("\nAttention shape analysis:")
    # print(f"Number of layers: {len(attention)}")
    # print(f"Attention shape per layer: {attention[0].shape}")
    # print(f"Final aggregated shape: {result.shape}")