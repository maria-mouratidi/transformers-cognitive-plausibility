import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_model import load_llama
from typing import List, Tuple, Dict

def word_to_tokens(encoded):

    tokens_mapping = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = [start, end-1]
            if len(tokens_mapping) == 0 or tokens_mapping[-1] != tokens:
                tokens_mapping.append(tokens)
    
    return tokens_mapping

def get_sentence_attention(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sentences: List[str]
) -> Tuple[List[List[str]], torch.Tensor]:

    # Tokenize with padding and truncation
    encoded = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    
    # Move inputs to model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            return_dict=True,
)
    
    attention_weights = outputs.attentions
    
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

   _, tokenizer = load_llama()
   print(word_to_tokens(tokenizer))
    
    
    # torch.cuda.empty_cache()
    # torch.set_default_device('cuda:0')

    # # Load model
    # model, tokenizer = load_model()
    
    # # Optional: Save model in safetensors format
    # #save_model(model, tokenizer, "/scratch/7982399/hf_cache")
    
    # sentences = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "She read the book that he recommended to them."
    # ]
    
    # # Get attention patterns
    # tokens, attention = get_sentence_attention(model, tokenizer, sentences)
    
    # # Aggregate attention
    # result = aggregate_attention(attention)
    
    # # Print results
    # print("\nAttention shape analysis:")
    # print(f"Number of layers: {len(attention)}")
    # print(f"Attention shape per layer: {attention[0].shape}")
    # print(f"Final aggregated shape: {result.shape}")