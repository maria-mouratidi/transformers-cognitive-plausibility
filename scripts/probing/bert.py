import torch
from scripts.probing.load_model import load_bert
from typing import List, Tuple
from materials.prompts import prompt_task2, prompt_task3
import re
import json

def encode_input(sentences: List[List[str]], tokenizer, task: str):
    """
    Encodes input sentences using BERT tokenizer.

    Args:
        sentences: List of pretokenized sentences (List[List[str]])
        tokenizer: The tokenizer to use
        task: 'none', 'task2', or 'task3'

    Returns:
        Tuple of (batch_encodings, processed_sentences, prompt_len)
    """
    if task == "none":
        prompt_words = []
        sentences_to_encode = [words for words in sentences]

    elif task == "task2":
        prompt_words = re.sub(r'[^\w\s]', '', prompt_task2).split()
        sentences_to_encode = [prompt_words + sentence for sentence in sentences]

    elif task == "task3":
        sentences_to_encode = []
        for item in sentences:
            sent, relation = item["sentence"], item["relation_type"]
            prompt_words = re.sub(r'[^\w\s]', '', prompt_task3.format(relation)).split()
            combined = prompt_words + sent
            sentences_to_encode.append(combined)
    else:
        raise ValueError(f"Invalid task: {task}")

    batch_encodings = tokenizer(
        sentences_to_encode,
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt",
        is_split_into_words=True  # crucial for mapping
    )

    return batch_encodings, sentences_to_encode, len(prompt_words)


def get_word_mappings(sentences: List[List[str]], batch_encodings, tokenizer) -> List[List[Tuple[str, int, List[str]]]]:
    """
    Maps each word to the number of tokens assigned to it.

    Args:
        sentences: List of original sentences as lists of strings
        batch_encodings: Tokenizer encodings with `is_split_into_words=True`
        tokenizer: Tokenizer used

    Returns:
        List of word mappings per sentence
    """
    word_mappings = []

    for sentence_idx in range(len(sentences)):
        word_ids = batch_encodings.word_ids(batch_index=sentence_idx)
        tokens = batch_encodings["input_ids"][sentence_idx]
        mapping = {}
        
        for token_idx, word_id in enumerate(word_ids):
            token_str = tokenizer.convert_ids_to_tokens(tokens[token_idx].item())
        
            if word_id is None:
                # Attach special tokens: CLS to first word, SEP to last
                if token_str == "[CLS]":
                    mapping[0] = [token_str]
                elif token_str == "[SEP]":
                    mapping[len(sentences[sentence_idx]) - 1].append(token_str)
            else:
                mapping.setdefault(word_id, []).append(token_str)

        word_mapping = []
        for idx, word in enumerate(sentences[sentence_idx]):
            token_list = mapping.get(idx, [])
            word_mapping.append((word, len(token_list), token_list))
        word_mappings.append(word_mapping)

    return word_mappings


def get_attention(model, encodings):
    """
    Get attention weights from a BERT model.

    Args:
        model: Pretrained model
        encodings: Tokenized inputs

    Returns:
        Attention tensor [layers, batch, heads, seq_len, seq_len]
    """
    device = next(model.parameters()).device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.inference_mode():
        output = model(**encodings, output_attentions=True)
        attention = torch.stack(output.attentions)

    return attention


def process_attention(attention: torch.Tensor, word_mappings: List[List[Tuple[str, int, List[str]]]], prompt_len: int, reduction: str = "mean") -> torch.Tensor:
    """
    Aggregate token-level attention into word-level, normalized attention.

    Args:
        attention: [layers, batch, heads, seq_len, seq_len]
        word_mappings: Word-token mappings
        prompt_len: Prompt word count
        reduction: 'mean' or 'max'

    Returns:
        Normalized word-level attention [layers, batch, max_words]
    """
    num_layers, batch_size, num_heads, seq_len, _ = attention.shape
    device = attention.device

    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = torch.zeros((num_layers, batch_size, num_heads, seq_len, max_words), device=device)

    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens, tokens) in enumerate(word_map):
            token_attentions = []
            token_offset = sum(w[1] for w in word_map[:word_idx])

            for n_token in range(num_tokens):
                token_idx = token_offset + n_token
                token_attention = attention[:, sentence_idx, :, :, token_idx]
                token_attentions.append(token_attention)

            if not token_attentions:
                continue

            token_attentions = torch.stack(token_attentions, dim=-1)

            if reduction == "mean":
                word_attention = token_attentions.mean(dim=-1)
            elif reduction == "max":
                word_attention = token_attentions.max(dim=-1).values
            else:
                raise ValueError("Reduction must be 'mean' or 'max'.")

            word_attentions[:, sentence_idx, :, :, word_idx] = word_attention

    word_avg = word_attentions.mean(dim=3)       # over sequence
    head_avg = word_avg.mean(dim=2)              # over heads
    head_avg = head_avg.squeeze(2)               # [layers, batch, max_words]

    attention_sum = head_avg.sum(dim=2, keepdim=True)
    normalized_attention = head_avg / (attention_sum + 1e-8)

    return normalized_attention[:, :, prompt_len:]


if __name__ == "__main__":
    task = "task2"  # Options: 'none', 'task2', 'task3'
    #model, tokenizer = load_bert()
    # subset = False  # For debugging

    # with open(f'materials/sentences_{task}.json', 'r') as f:
    #     sentences = json.load(f)

    # encodings, sentences_full, prompt_len = encode_input(sentences, tokenizer, task)
    # word_mappings = get_word_mappings(sentences_full, encodings, tokenizer)
    # attention = get_attention(model, encodings)
    # print("Attention shape:", attention.shape)

    # torch.save({
    #     'attention': attention,
    #     'input_ids': encodings['input_ids'],
    #     'word_mappings': word_mappings,
    #     'prompt_len': prompt_len
    # }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data_bert.pt")

    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data_bert.pt")
    attention_processed = process_attention(
        loaded_data['attention'],
        loaded_data['word_mappings'],
        loaded_data['prompt_len'],
        reduction="max"
    )

    torch.save({
        'attention_processed': attention_processed,
        'input_ids': loaded_data['input_ids'],
        'word_mappings': loaded_data['word_mappings'],
        'prompt_len': loaded_data['prompt_len']
    }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_processed_bert.pt")

    print("Shape:", attention_processed.shape)
