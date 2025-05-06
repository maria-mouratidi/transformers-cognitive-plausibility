import numpy as np
import math
import json
import wordfreq
import nltk
import pandas as pd
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
from scripts.probing.load_model import load_llama
from typing import List, Tuple

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_frequency(word: str) -> float:
    return wordfreq.zipf_frequency(word, 'en', wordlist='large')

def get_length(word: str) -> int:
    return len(word)

def get_role(word: str) -> str:
    return 'function' if word.lower() in stop_words else 'content'

def get_surprisals(
    model,
    input_ids: dict,
    word_mappings: List[List[Tuple[str, int]]],
    prompt_len: int,
    pad_id: int,
    device: str = "cpu"
) -> List[List[Tuple[str, float]]]:
    """
    Compute word-level surprisals from tokenized inputs and mappings.

    Returns:
        List of lists: each inner list contains (word, surprisal) for a sentence.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(input_ids).logits  # shape: [batch, seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)

    log2 = math.log(2)  # constant denominator
    surprisals = []

    for sent_idx, (word_map, input_id_row) in enumerate(zip(word_mappings, input_ids)):
        tokens = input_id_row.tolist()
        # calculate starting position after the prompt
        token_pos = sum(n for _, n in word_map[:prompt_len])
        sentence_surprisals = []

        for word, token_count in word_map[prompt_len:]:
            # compute surprisal for valid tokens in the word
            token_surprisals = []
            for pos in range(token_pos, token_pos + token_count):
                if tokens[pos] == pad_id:
                    continue
                # Guard against out-of-bounds (e.g. pos==0)
                if pos == 0:
                    token_surprisals.append(0.0)
                else:
                    token_log_prob = log_probs[sent_idx, pos - 1, tokens[pos]]
                    token_surprisals.append(-token_log_prob.item() / log2)
            word_surprisal = sum(token_surprisals)
            sentence_surprisals.append((word, word_surprisal))
            token_pos += token_count

        surprisals.append(sentence_surprisals)

    return surprisals

subset = False
if __name__ == "__main__":

    task = "task2"
    attention_method = "raw"
    # model_type = "causal" #'qa' for task3
    # model, tokenizer = load_llama(model_type=model_type)

    # Load the sentences
    with open(f'materials/sentences_{task}.json', 'r') as f:
        sentences = json.load(f)
    
    # Subset for testing
    if subset:
        print(f"Using subset of {subset} sentences")
        sentences = sentences[:subset]

    # loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/{attention_method}/attention_data.pt")
    # attention = loaded_data['attention']
    # word_mappings = loaded_data['word_mappings']
    # input_ids = loaded_data['input_ids']
    # prompt_len = loaded_data['prompt_len']

    # surprisals = get_surprisals(model, input_ids, word_mappings, prompt_len, tokenizer.pad_token_id)   
    # with open(f"outputs/{task}/{attention_method}/surprisals.json", "w") as f:
    #     json.dump(surprisals, f)

    with open(f"outputs/{task}/{attention_method}/surprisals.json", "r") as f:
        surprisals = json.load(f)
    
    features = []
    for sentence_idx, sentence in enumerate(sentences):
        sentence_surprisals = surprisals[sentence_idx]
        # Extract features for each token
        for word_idx, (word, (_, surprisal)) in enumerate(zip(sentence, sentence_surprisals)):
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
    df.to_csv('materials/text_features.csv', index=False)



