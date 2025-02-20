import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import os
import gc

def load_model(
    model_id: str = "meta-llama/Llama-3.1-8B",
    cache_dir: str = '/scratch/7982399/hf_cache',
    local_path: str = '/scratch/7982399/hf_cache') -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path if local_path else model_id, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",)
        #device_map='auto')#.to(device)
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Compile model if requested and available
    if hasattr(torch, 'compile') and device == 'cuda':
        model = torch.compile(model)
    
    model.eval()
    return model, tokenizer

def save_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def get_sentence_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: List[str]) -> Tuple[List[List[str]], torch.Tensor]:

    # Tokenize with padding and truncation
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    
    # Move inputs to model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            return_dict=True,
)
    
    attention_weights = outputs.attentions
    
    # Convert to tokens
    tokens_batch = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs["input_ids"].tolist()]
    
    return tokens_batch, attention_weights


def aggregate_attention(attention_weights: torch.Tensor) -> torch.Tensor:
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
    torch.set_default_device('cuda:0')

    # Load model
    model, tokenizer = load_model()
    
    # Optional: Save model in safetensors format
    #save_model(model, tokenizer, "/scratch/7982399/hf_cache")
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She read the book that he recommended to them."
    ]
    
    # Get attention patterns
    tokens, attention = get_sentence_attention(model, tokenizer, sentences)
    
    # Aggregate attention
    result = aggregate_attention(attention)
    
    # Print results
    print("\nAttention shape analysis:")
    print(f"Number of layers: {len(attention)}")
    print(f"Attention shape per layer: {attention[0].shape}")
    print(f"Final aggregated shape: {result.shape}")