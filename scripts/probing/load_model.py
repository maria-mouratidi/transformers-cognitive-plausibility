import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict

def load_llama(
    model_id: str = "meta-llama/Llama-3.1-8B",
    cache_dir: str = '/scratch/7982399/hf_cache',
    local_path: str = '/scratch/7982399/hf_cache') -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path if local_path else model_id, cache_dir=cache_dir)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
        #device_map='auto').to(device)
    
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
