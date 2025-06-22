import numpy as np
import math
import json
import wordfreq
import nltk
import pandas as pd
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
from scripts.probing.load_model import load_bert, load_llama
from typing import List, Tuple
from tqdm import tqdm

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_frequency(word: str) -> float:
    return wordfreq.zipf_frequency(word, 'en', wordlist='large')

def get_length(word: str) -> int:
    return len(word)

def get_role(word: str) -> str:
    return 'function' if word.lower() in stop_words else 'content'

def get_surprisals(
    model,
    input_ids: torch.Tensor,
    word_mappings: list,
    prompt_len: int,
    pad_id: int,
    model_name: str,
    tokenizer=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list:
    model.to(device)
    model.eval()
    log2 = math.log(2)
    surprisals = []

    if model_name == "bert":
        # BERT: Masked LM surprisals
        for sent_idx, (word_map, input_id_row) in tqdm(
            enumerate(zip(word_mappings, input_ids)), total=len(word_mappings), desc="BERT sentences"
        ):
            tokens = input_id_row.tolist()
            sentence_surprisals = []
            token_pos = sum(n for _, n, _ in word_map[:prompt_len])
            for word, num_tokens, word_tokens in word_map[prompt_len:]:
                word_surprisal = 0.0
                for i, tok_id in enumerate(word_tokens):
                    if isinstance(tok_id, str):
                        tok_id = tokenizer.convert_tokens_to_ids(tok_id)
                    pos = token_pos + i
                    if tokens[pos] == pad_id:
                        continue
                    masked = input_id_row.clone()
                    masked[pos] = tokenizer.mask_token_id
                    with torch.no_grad():
                        outputs = model(masked.unsqueeze(0).to(device))
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_log_prob = log_probs[0, pos, tok_id]
                        word_surprisal += -token_log_prob.item() / log2
                sentence_surprisals.append((word, word_surprisal))
                token_pos += num_tokens
            surprisals.append(sentence_surprisals)
    elif model_name == "llama":
        # LLaMA: Causal LM surprisals
        with torch.no_grad():
            logits = model(input_ids).logits  # [batch, seq_len, vocab]
            log_probs = F.log_softmax(logits, dim=-1)
        for sent_idx, (word_map, input_id_row) in tqdm(
            enumerate(zip(word_mappings, input_ids)), total=len(word_mappings), desc="LLaMA sentences"
        ):
            tokens = input_id_row.tolist()
            token_pos = sum(n for _, n, _ in word_map[:prompt_len])
            sentence_surprisals = []
            for word, num_tokens, word_tokens in word_map[prompt_len:]:
                token_surprisals = []
                for i, tok_id in enumerate(word_tokens):
                    if isinstance(tok_id, str):
                        tok_id = tokenizer.convert_tokens_to_ids(tok_id)
                    pos = token_pos + i
                    if tokens[pos] == pad_id:
                        continue
                    if pos == 0:
                        token_surprisals.append(0.0)
                    else:
                        token_log_prob = log_probs[sent_idx, pos - 1, tok_id]
                        token_surprisals.append(-token_log_prob.item() / log2)
                word_surprisal = sum(token_surprisals)
                sentence_surprisals.append((word, word_surprisal))
                token_pos += num_tokens
            surprisals.append(sentence_surprisals)
    return surprisals

subset = False

if __name__ == "__main__":
    tasks = ["task2", "task3"]
    models = ["bert", "llama"]
    attention_method = "raw"
    
    for model_name in models:
        print(f"Processing text features for model: {model_name}")
        # Load model and tokenizer
        if model_name == "llama":
            model, tokenizer = load_llama("causal")
        elif model_name == "bert":
            model, tokenizer = load_bert(maskedLM=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        for task in tasks:
            # Load the sentences
            print(f"Calculating text feature for model: {model_name} task: {task}")
            with open(f'materials/sentences_{task}.json', 'r') as f:
                sentences = json.load(f)


            # Load attention data (for input_ids, word_mappings, prompt_len)
            loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{attention_method}/{task}/{model_name}/attention_data.pt")
            attention = loaded_data['attention']
            word_mappings = loaded_data['word_mappings']
            input_ids = loaded_data['input_ids']
            prompt_len = loaded_data['prompt_len']

            # Compute surprisals
            surprisals = get_surprisals(model, input_ids, word_mappings, prompt_len, tokenizer.pad_token_id, model_name, tokenizer)
            # Save surprisals for reference
            with open(f"outputs/{attention_method}/{task}/{model_name}/surprisals.json", "w") as f:
                json.dump(surprisals, f)

            # Extract features
            features = []
            for sentence_idx, sentence in enumerate(sentences):
                sentence_surprisals = surprisals[sentence_idx]
                for word_idx, (word, surprisal) in enumerate(sentence_surprisals):
                    frequency = get_frequency(word)
                    length = get_length(word)
                    role = get_role(word)
                    features.append({
                        'Sent_ID': sentence_idx,
                        'Word_ID': word_idx,
                        'Word': word,
                        'frequency': frequency,
                        'length': length,
                        'surprisal': surprisal,
                        'role': role
                    })

            df = pd.DataFrame(features)
            df.sort_values(['Sent_ID', 'Word_ID'], inplace=True)
            df.to_csv(f'materials/text_features_{task}_{model_name}.csv', index=False)



