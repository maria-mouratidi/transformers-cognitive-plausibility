import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForQuestionAnswering, BertForMaskedLM
from typing import List, Tuple, Dict, Union, Literal
transformers.logging.set_verbosity_error()

hf_token = "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"

def load_llama(
    model_type: str,
    model_id: str = "meta-llama/Llama-3.1-8B",
    cache_dir: str = '/scratch/7982399/hf_cache/llama',
    local_path: str = '/scratch/7982399/hf_cache/llama') -> Tuple[Union[AutoModel, AutoModelForCausalLM, AutoModelForQuestionAnswering], AutoTokenizer]:
    """
    Load a model and tokenizer with specified architecture type.
    
    Args:
        model_id: Hugging Face model ID
        cache_dir: Directory to cache the model
        local_path: Local path to the model if already downloaded
        task: Task type (between "task2" and "task3) to decide which model head to load
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model_class = AutoModelForCausalLM

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir, use_auth_token=hf_token)
    
    # Set tokenizer padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    
    # Load model
    model = model_class.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
        #device_map="auto", )# Automatically map to GPU if available


    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    return model, tokenizer

def load_bert(
    model_id: str = "bert-base-uncased",
    cache_dir: str = '/scratch/7982399/hf_cache/bert',
    local_path: str = '/scratch/7982399/hf_cache/bert',
    maskedLM: bool = False) -> Tuple[AutoModel, AutoTokenizer]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if maskedLM:
        model_class = BertForMaskedLM
    else:
        model_class = AutoModel
    model = model_class.from_pretrained(
        local_path if local_path else model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,  # BERT is typically FP32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(local_path if local_path else model_id, cache_dir=cache_dir)
    model.eval()
    return model, tokenizer


def save_model(model: AutoModel, tokenizer: AutoTokenizer, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")