import numpy as np
import torch
import networkx as nx
from networkx.algorithms import flow
import os
from tqdm import tqdm
from typing import List, Tuple

def calculate_flow(
    attention_tensor,               # <-- now a tensor, not a dict
    batch,
    output_token=-1,
    show=False,
    plot=True,
):
    layers, batches, heads, seq_len, _ = attention_tensor.shape
    num_input_tokens = seq_len - 1
    num_output_tokens = 1

    g = nx.DiGraph()

    s = (-1, int(num_input_tokens + num_output_tokens / 2)) # source node before the first layer (centered, just to match graph theory conventional figures)
    t = (layers + 2, int(num_input_tokens + num_output_tokens / 2)) # sink node after the last layer (centered, just to match graph theory conventional figures)
    g.add_node(s)
    g.add_node(t)

    attention_tensor = attention_tensor.mean(dim=2)  # Average over heads
    # New attention shape: [layers, batches, seq_len, seq_len]
    
    attn = attention_tensor[:, batch, :, :]  # Select batch
    # New attention shape: [layers, seq_len, seq_len]

    for x in range(layers):
        for y in range(seq_len):
            for z in range(y, seq_len): # causal attention to past tokens
                attn_weight = float(attn[x, z, y]) # attention from z to y
                g.add_edge((x, y), (x + 1, z), capacity=attn_weight)

    g.add_edge((layers, output_token + num_input_tokens), t, capacity=np.inf)

    flow_scores = []
    for x in range(num_input_tokens): # max flow: input token --> target
        flow_value, _ = nx.maximum_flow(g, (0, x), t, flow_func=flow.edmonds_karp)
        flow_scores.append(flow_value)

    for x in range(num_input_tokens, num_input_tokens + num_output_tokens): # max flow: output token --> target
        flow_value, _ = nx.maximum_flow(g, (0, x), t, flow_func=flow.edmonds_karp)
        flow_scores.append(flow_value)

    for x in range(num_input_tokens + output_token): # normalize auto-regression
        flow_scores[x] *= 1 / (2 + num_input_tokens + output_token - x) # decay factor for early token bias
    
    return torch.Tensor(flow_scores).unsqueeze(0)  # Add batch dimension

def process_attention_flow(attention_flow: torch.Tensor, word_mappings: List[List[Tuple[str, int]]],
                           prompt_len: int, reduction: str = "mean") -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights for attention flow.
    Expects attention tensor of shape [batch_size, seq_len].
    """
    batch_size, seq_len, = attention_flow.shape
    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = torch.zeros((batch_size, max_words), dtype=torch.float32)
    
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens, tokens) in enumerate(word_map):
            token_attentions = []
            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                token_idx = sum(token[1] for token in prev_tokens) + n_token
                token_attention = attention_flow[sentence_idx, token_idx] 
                token_attentions.append(token_attention)

            token_attentions = torch.stack(token_attentions)  # [seq_len, num_tokens]
            if reduction == "mean":
                word_attention = token_attentions.mean(dim=0)  # [seq_len]
            elif reduction == "max":
                word_attention = token_attentions.max(dim=0)[0]
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")

            word_attentions[sentence_idx, word_idx] = word_attention

    word_attentions /= word_attentions.sum(dim=1, keepdim=True)  # Normalize attention per sentence
    # Remove prompt words
    return word_attentions[:, prompt_len:]  # [batch_size, max_words - prompt_len]



if __name__ == "__main__":
    # Load your data
    task = "task3"
    raw_data = torch.load(f"/scratch/7982399/thesis/outputs/raw/{task}/llama/attention_data.pt")
    attn = raw_data['attention']  # tensor of shape [layers, batches, heads, seq_len, seq_len]
    word_mappings = raw_data['word_mappings']  # List of (token, position) tuples per batch
    prompt_len = raw_data['prompt_len']  # Length of the prompt in tokens
    
    flow_dir = f"/scratch/7982399/thesis/outputs/flow/{task}/llama"
    os.makedirs(flow_dir, exist_ok=True)

    all_flows = []

    for batch in tqdm(range(attn.shape[1]), desc="Processing Batches"):
        # Example usage
        result = calculate_flow(
            attention_tensor=attn, 
            batch=batch,
        )
        #batch_dir = os.path.join(flow_dir, f"batch_{batch}.pt")  # Save the result for each batch for safety
        #torch.save(result, batch_dir)
        all_flows.append(result)

    stacked_results = torch.cat(all_flows, dim=0)  # Concatenate along the batch dimension
    stacked_dir = os.path.join(flow_dir, "attention_flow.pt")
    torch.save(stacked_results, stacked_dir)

    attention_flow = torch.load(stacked_dir)
    flow_processed = process_attention_flow(attention_flow, word_mappings, prompt_len=prompt_len, reduction="max")
    # Save the processed flow
    processed_dir = os.path.join(flow_dir, "attention_flow_processed.pt")
    torch.save(flow_processed, processed_dir)
    print(f"Processed attention flow saved to {processed_dir}")