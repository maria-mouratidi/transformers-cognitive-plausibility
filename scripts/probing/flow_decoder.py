import numpy as np
import torch
import networkx as nx
from networkx.algorithms import flow
import os
from tqdm import tqdm

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


if __name__ == "__main__":
    # Load your data
    task = "task2"
    raw_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")
    attn = raw_data['attention']  # tensor of shape [layers, batches, heads, seq_len, seq_len]
    # word_mappings = loaded_data['word_mappings']  # List of (token, position) tuples per batch

    flow_dir = f"/scratch/7982399/thesis/outputs/{task}/flow"
    os.makedirs(flow_dir, exist_ok=True)

    all_flows = []

    for batch in tqdm(6, range(attn.shape[1]), desc="Processing Batches"):
        # Example usage
        result = calculate_flow(
            attention_tensor=attn, 
            batch=batch,
        )
        batch_dir = os.path.join(flow_dir, f"batch_{batch}.pt")  # Save the result for each batch for safety
        torch.save(result, batch_dir)
        all_flows.append(result)

    stacked_results = torch.cat(all_flows, dim=0)  # Concatenate along the batch dimension
    stacked_dir = os.path.join(flow_dir, "attention_flow_decoder.pt")
    torch.save(stacked_results, stacked_dir)