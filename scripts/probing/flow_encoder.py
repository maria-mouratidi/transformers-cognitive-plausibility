import numpy as np
import torch
import networkx as nx
from networkx.algorithms import flow
import os
from tqdm import tqdm
from typing import List, Tuple

def calculate_encoder_flow(
    attention_tensor: torch.Tensor,
    batch: int,
    output_token: int = -1,
) -> torch.Tensor:
    """
    Calculates attention flow for a BERT-style encoder-only model.

    attention_tensor: [layers, batches, heads, seq_len, seq_len]
    output_token: Token index to connect to sink; if -1, connect all tokens.
    """
    layers, batches, heads, seq_len, _ = attention_tensor.shape

    g = nx.DiGraph()
    s = (-1, seq_len // 2)
    t = (layers + 2, seq_len // 2)
    g.add_node(s)
    g.add_node(t)

    # Average over heads
    # Ensure the attention tensor is on the CPU
    attention = attention_tensor.mean(dim=2)  # [layers, batches, seq_len, seq_len]
    attn = attention[:, batch, :, :]  # [layers, seq_len, seq_len]

    # Add edges layer to layer
    for x in range(layers):
        for y in range(seq_len):
            for z in range(seq_len):
                weight = float(attn[x, z, y])  # attention from z -> y
                g.add_edge((x, y), (x + 1, z), capacity=weight)

    # Connect output tokens to sink
    if output_token >= 0:
        g.add_edge((layers, output_token), t, capacity=np.inf)
    else:
        for i in range(seq_len):
            g.add_edge((layers, i), t, capacity=np.inf)

    # Compute flow from input tokens to sink
    flow_scores = []
    for x in range(seq_len):
        flow_value, _ = nx.maximum_flow(g, (0, x), t, flow_func=flow.edmonds_karp)
        flow_scores.append(flow_value)

    return torch.Tensor(flow_scores).unsqueeze(0)  # [1, seq_len]


def process_attention_flow(attention_flow: torch.Tensor, word_mappings: List[List[Tuple[str, int]]],
                           prompt_len: int, reduction: str = "mean") -> torch.Tensor:
    """
    Token-to-word aggregation for encoder flow.
    """
    # Ensure attention_flow is on CPU
    batch_size, seq_len = attention_flow.shape
    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = torch.zeros((batch_size, max_words), dtype=torch.float32)

    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens, tokens) in enumerate(word_map):
            token_attentions = []
            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                token_idx = sum(tok[1] for tok in prev_tokens) + n_token
                token_attentions.append(attention_flow[sentence_idx, token_idx])

            token_attentions = torch.stack(token_attentions)
            if reduction == "mean":
                word_attention = token_attentions.mean()
            elif reduction == "max":
                word_attention = token_attentions.max()
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")

            word_attentions[sentence_idx, word_idx] = word_attention

    word_attentions /= word_attentions.sum(dim=1, keepdim=True)
    return word_attentions[:, prompt_len:]


if __name__ == "__main__":

    task, model_name = "task2", "bert"
    raw_data = torch.load(f"/scratch/7982399/thesis/outputs/raw/{task}/{model_name}/attention_data.pt")
    attn = raw_data['attention']
    word_mappings = raw_data['word_mappings']
    prompt_len = raw_data['prompt_len']
    num_sentences = attn.shape[1]
    flow_dir = f"/scratch/7982399/thesis/outputs/flow/{task}/{model_name}"
    # os.makedirs(flow_dir, exist_ok=True)

    all_flows = []
    for batch in tqdm(range(num_sentences), desc="Processing Batches"):
        result = calculate_encoder_flow(attention_tensor=attn, batch=batch)
        all_flows.append(result)

    stacked_results = torch.cat(all_flows, dim=0)
    torch.save(stacked_results, os.path.join(flow_dir, "attention_flow.pt"))
    
    attention_flow = torch.load(os.path.join(flow_dir, "attention_flow.pt"))
    flow_processed = process_attention_flow(attention_flow, [word_mappings[0]], prompt_len=prompt_len, reduction="max")
    torch.save(flow_processed, os.path.join(flow_dir, "attention_flow_processed.pt"))
    attention_flow_processed = torch.load(os.path.join(flow_dir, "attention_flow_processed.pt"))
    
    print(f"Attention flow processed shape: {attention_flow_processed.shape}")

