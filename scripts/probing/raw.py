import numpy as np
import torch
import scipy.special
from transformers import LlamaTokenizer, LlamaForCausalLM


def load_model(model_id="meta-llama/Llama-2-7b-hf"):

    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,  # half precision for efficiency
        output_attentions=True
    )
    model.eval()  # evaluation mode
    return model, tokenizer

def get_sentence_attention(model, tokenizer, sentence):
    """
    Extract attention for a given sentence
    
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    
    with torch.no_grad(): # no training setting
        outputs = model(input_ids=inputs["input_ids"], 
                       output_attentions=True,
                       return_dict=True)
    
    attention = outputs.attentions
    
    # Convert input IDs to tokens
    input_id_list = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    return tokens, attention

model, tokenizer = load_model()
sentence = "The quick brown fox jumps over the lazy dog."
print(get_sentence_attention(model, tokenizer, sentence))


    