import networkx as nx
import numpy as np
import torch
from typing import List, Tuple, Dict
import multiprocessing as mp
from tqdm import tqdm
from scripts.probing.load_model import *
from scripts.probing.raw import *

def build_attention_graph(att: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Build a graph-compatible adjacency matrix from multi-layer attention.
    
    Args:
        att: [num_layers, seq_len, seq_len] averaged and normalized attention

    Returns:
        adj_matrix: [num_nodes, num_nodes]
        node_map: {"L{layer}_{token_idx}": graph_node_idx}
    """
    num_layers, seq_len, _ = att.shape
    num_nodes = (num_layers + 1) * seq_len
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    node_map = {f"L{l}_{i}": l * seq_len + i for l in range(num_layers + 1) for i in range(seq_len)}

    for l in range(num_layers):
        for i in range(seq_len):
            for j in range(i + 1):  # causal mask: j ≤ i
                src = node_map[f"L{l}_{i}"]
                tgt = node_map[f"L{l+1}_{j}"]
                adj_matrix[src, tgt] = att[l, i, j]
    
    return adj_matrix, node_map

def compute_single_node_flow(args) -> Tuple[int, np.ndarray]:
    """
    Computes flow values from all input_nodes to a single output_node using nx.maximum_flow.

    Returns:
        output_token_index: The index of the output token (0-based).
        relevance_row: Array of shape [seq_len] with normalized flow values.
    """
    G, labels_to_index, input_nodes, output_node, seq_len = args
    u = labels_to_index[output_node]
    layer = int(output_node.split("_")[0][1:]) - 1
    flows = np.zeros(seq_len, dtype=np.float32)

    for inp_node in input_nodes:
        v = labels_to_index[inp_node]
        flow_dict = nx.maximum_flow(G, v, u)[1]
        flow_value = flow_dict.get(v, {}).get(u, 0.0)
        flows[v % seq_len] = flow_value

    flow_norm = flows / flows.sum()

    return flow_norm


def compute_node_flow_parallel(
    G: nx.DiGraph,
    labels_to_index: Dict[str, int],
    input_nodes: List[str],
    output_nodes: List[str],
    seq_len: int,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Computes flow values using parallel processing for each output node.

    Returns:
        flow_matrix: Array [num_output_nodes, seq_len] with relevance values.
    """
    args = [(G, labels_to_index, input_nodes, out_node, seq_len) for out_node in output_nodes]
    with mp.Pool(num_workers) as pool:
        results = pool.map(compute_single_node_flow, args)

    flow_matrix = np.stack(results, axis=0)  # [num_output_nodes, seq_len]
    return flow_matrix

def get_flow_relevance(raw_attention: np.ndarray, input_ids: List[List[str]]) -> np.ndarray:
    """
    Compute attention flow relevance across layers and batches.
    
    Args:
        raw_attention: [num_layers, batch_size, num_heads, seq_len, seq_len]
        input_ids: token IDs per batch (used for length only)

    Returns:
        flow_relevance: [num_layers, batch_size, seq_len, seq_len]
    """
    num_layers, batch_size, num_heads, seq_len, _ = raw_attention.shape
    flow_relevance = np.zeros((num_layers, batch_size, seq_len, seq_len), dtype=np.float32)

    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    identity = np.eye(seq_len, dtype=np.float32)

    for b in range(batch_size):
        att = raw_attention[:, b].mean(axis=0)  # [num_layers, seq_len, seq_len]
        att *= causal_mask[np.newaxis, :, :]
        att += identity[np.newaxis, :, :]  # Add identity to avoid zero flow
        att /= att.sum(axis=-1, keepdims=True)

        adj_matrix, node_map = build_attention_graph(att)
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
        nx.set_edge_attributes(G, {e: {"capacity": adj_matrix[e]} for e in G.edges})

        input_nodes = [f"L0_{i}" for i in range(seq_len)]

        for layer in tqdm(range(num_layers), desc="Processing layers"):
            output_nodes = [f"L{layer + 1}_{j}" for j in range(seq_len)]
            flow_matrix = compute_node_flow_parallel(G, node_map, input_nodes, output_nodes, seq_len, num_workers=4)
            flow_relevance[layer, b] = flow_matrix.T  # transpose to match shape [seq_len, seq_len]

    return flow_relevance  # [num_layers, batch_size, seq_len, seq_len]


def process_attention_flow(attention: np.ndarray, word_mappings: List[List[Tuple[str, int]]],
                           prompt_len: int,reduction: str = "mean") -> np.ndarray:
    """
    Extract word-level attention from token-level attention weights for attention flow (NumPy version).

    Args:
        attention: NumPy array [num_layers, batch_size, seq_len, seq_len]
        word_mappings: List of token counts for each word in each sentence
        prompt_len: Length of the prompt, used to filter out prompt tokens
        reduction: Reduction method, either "mean" (average) or "max" (max of tokens per word)

    Returns:
        Word attention NumPy array [num_layers, batch_size, seq_len, max_words]
    """
    print("initial attention shape:", attention.shape)
    num_layers, batch_size, seq_len, _ = attention.shape

    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = np.zeros((num_layers, batch_size, seq_len, max_words), dtype=np.float32)

    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens) in enumerate(word_map):
            token_attentions = []

            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                token_idx = sum(token[1] for token in prev_tokens) + n_token
                token_attention = attention[:, sentence_idx, :, token_idx]  # [num_layers, seq_len]
                token_attentions.append(token_attention)

            token_attentions = np.stack(token_attentions, axis=-1)  # [num_layers, seq_len, num_tokens]

            # Apply reduction over tokens
            if reduction == "mean":
                word_attention = np.mean(token_attentions, axis=-1)  # [num_layers, seq_len]
            elif reduction == "max":
                word_attention = np.max(token_attentions, axis=-1)  # [num_layers, seq_len]
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")
            
            word_attentions[:, sentence_idx, :, word_idx] = word_attention

    word_attentions = np.mean(word_attentions, axis=2)  # [num_layers, batch_size, seq_len]  

    return word_attentions[:, :, prompt_len:]  # Remove prompt words
 
subset = False
if __name__ == "__main__":

    # model_type = "causal" #'qa' for task3
    # model, tokenizer = load_llama(model_type=model_type)
    task = "task2"

    # # Load the sentences
    # with open('materials/sentences.json', 'r') as f:
    #     sentences = json.load(f)
    
    # # Subset for testing
    # if subset:
    #     print(f"Using subset of {subset} sentences")
    #     sentences = sentences[:subset]

    # encodings, word_mappings, prompt_len = encode_input(sentences, tokenizer, task)

    # attention = get_attention(model, encodings) # [num_layers, batch_size, num_heads, seq_len, seq_len]

    # torch.save({
    #     'attention': attention,
    #     'word_mappings': word_mappings,
    #     'prompt_len': prompt_len,
    #     'input_ids': encodings['input_ids'],
    # }, f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

    # Load the attention data
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_subset_data.pt")

    # Extract raw attention over the input
    attention_tensor = loaded_data['attention']  # [num_layers, batch_size, num_heads, seq_len, seq_len]
    input_ids = loaded_data['input_ids']  # [batch_size, seq_len]
    word_mappings = loaded_data['word_mappings']  # List of word mappings
    prompt_len = loaded_data['prompt_len']  # Length of the prompt

    print("Raw attention shape: ", attention_tensor.shape)
    
    # Compute flow relevance for all layers and batches
    print("Computing flow relevance scores...")

    all_examples_flow_relevance = get_flow_relevance(attention_tensor, input_ids)
    #print("Flow relevance: ", all_examples_flow_relevance[0,0,0])

    # Save the results
    np.save(f'outputs/{task}/flow/attention_flow_subset.npy', all_examples_flow_relevance)

    # # Load the saved attention flow data
    # attention_flow = np.load('outputs/task2/flow/attention_flow.npy')

    # # Print the type and shape of the loaded data
    # print(attention_flow.shape)

    # # Process the attention flow data
    # attention_processed = process_attention_flow(attention_flow, word_mappings, prompt_len)
    # print("Processed attention shape: ", attention_processed.shape)
    # np.save(f'outputs/{task}/flow/attention_processed.npy', attention_processed)

    #TODO: refactor to use torch instead of numpy. or the opposite for raw attn
    #DEBUG: attention flow gets all 0s tensor, problem is in after row 105