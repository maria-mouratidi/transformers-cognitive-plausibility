import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_model import load_llama
from typing import List, Tuple, Dict
from collections import defaultdict

def get_word_offsets(sentences: List[str]) -> List[List[Tuple[str, int, int]]]:
    result = []
    
    for sentence in sentences:
        offsets = []
        current_offset = 0
        
        for word in sentence.split():
            start_offset = current_offset
            end_offset = start_offset + len(word)
            offsets.append((word, start_offset, end_offset))
            current_offset = end_offset
            
        result.append(offsets)
        

def map_tokens_to_words(
    word_offsets: List[List[Tuple[str, int, int]]],
    token_offsets: List[List[Tuple[int, int]]]
) -> Dict[int, Dict[str, List[int]]]:
    """Maps words to their corresponding token indices."""
    
    mapping = defaultdict(lambda: defaultdict(list))
    
    for sent_id, (sent_words, sent_tokens) in enumerate(zip(word_offsets, token_offsets)):
        for token_idx, (token_start, token_end) in enumerate(sent_tokens):
            # Skip empty tokens
            if token_start == token_end:
                continue
                
            # Find which word this token belongs to
            for word, word_start, word_end in sent_words:
                if word_start <= token_start and token_end <= word_end:
                    mapping[sent_id][word].append(token_idx)
                    break
    
    return mapping

def get_sentence_attention(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sentences: List[str]
) -> Tuple[List[List[str]], torch.Tensor]:

    # Tokenize with padding and truncation
    encoded = tokenizer(sentences, return_tensors="pt", return_offsets_mapping = True)
    # Move inputs to model's device
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    print(encoded)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**encoded,
            output_attentions=True,
            return_dict=True,
            )
    
    attention_weights = outputs.attentions
    word_mappings = []
    word_attentions = []

    #TODO: ask claude whether this can be done more efficiently.
    for sentence_idx, word_map in enumerate(word_mappings):
        print("Processing sentence: ", sentence_idx)
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

    # torch.cuda.empty_cache()
    # torch.set_default_device('cuda:0')

    model, tokenizer = load_llama()
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She read the book that he recommended to them."
    ]
    
    # Process sentences
    encoded = tokenizer(sentences, return_tensors="pt", return_offsets_mapping=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    
    # Get offsets and create mapping
    word_offsets = get_word_offsets(sentences)
    token_offsets = encoded['offset_mapping'].tolist()
    mapping = map_tokens_to_words(word_offsets, token_offsets)
    
    # Print results in a clean format
    for sent_id, word_dict in mapping.items():
        print(f"Sentence {sent_id}: {sentences[sent_id]}")
        print("Word to token mappings:")
        for word, tokens in word_dict.items():
            print(f"  '{word}': tokens {tokens}")
        print()
    # # Get attention patterns
    # print(get_sentence_attention(model, tokenizer, sentences))
    
    # # Aggregate attention
    # result = aggregate_attention(attention)
    
    # # Print results
    # print("\nAttention shape analysis:")
    # print(f"Number of layers: {len(attention)}")
    # print(f"Attention shape per layer: {attention[0].shape}")
    # print(f"Final aggregated shape: {result.shape}")