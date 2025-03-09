import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForQuestionAnswering
from typing import List, Tuple, Dict, Union, Literal

transformers.logging.set_verbosity_error()

def load_llama(
    model_id: str = "meta-llama/Llama-3.1-8B",
    cache_dir: str = '/scratch/7982399/hf_cache',
    local_path: str = '/scratch/7982399/hf_cache',
    model_type: Literal["base", "causal", "qa"] = "base") -> Tuple[Union[AutoModel, AutoModelForCausalLM, AutoModelForQuestionAnswering], AutoTokenizer]:
    """
    Load a model and tokenizer with specified architecture type.
    
    Args:
        model_id: Hugging Face model ID
        cache_dir: Directory to cache the model
        local_path: Local path to the model if already downloaded
        model_type: Type of model to load - "base" (AutoModel), "causal" (AutoModelForCausalLM), 
                   or "qa" (AutoModelForQuestionAnswering)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir)
    
    # Load the appropriate model type based on model_type parameter
    if model_type == "causal":
        model_class = AutoModelForCausalLM
    elif model_type == "qa":
        model_class = AutoModelForQuestionAnswering
    else:  # Default to base model
        model_class = AutoModel
    
    # Load model
    model = model_class.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True)#.to(device)
    
    model.eval()
    return model, tokenizer

def save_model(model: AutoModel, tokenizer: AutoTokenizer, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")