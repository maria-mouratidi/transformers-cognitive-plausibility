import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_id="meta-llama/Llama-3.1-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', cache_dir='/scratch/7982399/hf_cache', attn_implementation = "eager")
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
    # shape: Layer x Sentence x Attention head x Toke nA attention to x Token B
    
    # Convert input IDs to tokens
    input_id_list = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    return tokens, attention

model, tokenizer = load_model()
sentence = "The quick brown fox jumps over the lazy dog."
tokens, attention = get_sentence_attention(model, tokenizer, sentence)
print(tokens, attention[0].shape)


    