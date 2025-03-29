import networkx as nx
import numpy as np
import json
import torch
from typing import List, Tuple
from scripts.probing.load_model import *
from scripts.probing.raw import *

def get_adjmat(attention_matrix: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Converts an attention matrix into an adjacency matrix for graph processing.
    
    Args:
        attention_matrix: Array of shape [seq_len, seq_len], representing attention for a single layer.
        input_tokens: List of tokenized words corresponding to the sequence.
    
    Returns:
        adj_mat: Array of shape [seq_len, seq_len], representing the adjacency matrix.
        labels_to_index: Dictionary mapping token indices to graph node indices.
    """
    seq_len = attention_matrix.shape[0]
    
    # Ensure the attention matrix has correct shape
    assert attention_matrix.shape == (seq_len, seq_len), "Attention matrix must be square."
    
    # Initialize adjacency matrix
    adj_mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    # Create mapping from token indices to graph node indices
    labels_to_index = {i: i for i in range(seq_len)}

    # Fill adjacency matrix with attention values
    np.copyto(adj_mat, attention_matrix)

    return adj_mat, labels_to_index



def compute_node_flow(G: nx.DiGraph, labels_to_index: dict, input_nodes: List[str], output_nodes: List[str], seq_len: int) -> np.ndarray:
    """
    Computes node-to-node flow values based on maximum flow.

    Args:
        G: A directed graph representing token attention.
        labels_to_index: Dictionary mapping (layer, token) to unique node indices.
        input_nodes: List of input tokens (nodes).
        output_nodes: List of output tokens (nodes).
        seq_len: Length of the tokenized sentence.

    Returns:
        flow_values: A NumPy array of shape [seq_len, seq_len] representing attention flow.
    """
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))

    for key in output_nodes:
        if key not in input_nodes:
            current_layer = labels_to_index[key] // seq_len
            prev_layer = current_layer - 1
            u = labels_to_index[key]

            for inp_node in input_nodes:
                v = labels_to_index[inp_node]

                # Compute max flow between input token (v) and output token (u)
                flow_value = nx.maximum_flow_value(G, v, u)

                # Store normalized flow values
                flow_values[u][prev_layer * seq_len + v] = flow_value

            # Normalize per token
            if flow_values[u].sum() > 0:
                flow_values[u] /= flow_values[u].sum()

    return flow_values


def compute_flow_relevance(full_att_mat: np.ndarray) -> np.ndarray:
    """
    Computes flow relevance scores for all words in a sentence across all layers and batches.
    
    Args:
        full_att_mat: Attention matrices (shape: [num_layers, batch_size, num_heads, seq_len, seq_len])
    
    Returns:
        A NumPy array of shape [num_layers, batch_size, seq_len, seq_len] representing flow relevance scores.
    """
    num_layers, batch_size, num_heads, seq_len, _ = full_att_mat.shape

    # Initialize a tensor to store flow relevance scores
    all_layers_flow_relevance = np.zeros((num_layers, batch_size, seq_len, seq_len), dtype=np.float32)

    for layer in range(num_layers):
        for batch_idx in range(batch_size):
            # Extract attention for the current layer and batch
            batch_attention = full_att_mat[layer, batch_idx, :, :, :]  # Shape: [num_heads, seq_len, seq_len]

            # Aggregate attention across heads and normalize
            res_att_mat = batch_attention.sum(axis=0) / num_heads
            res_att_mat += np.eye(seq_len, dtype=np.float32)  
            res_att_mat /= res_att_mat.sum(axis=-1, keepdims=True)

            # Convert attention to graph format
            res_adj_mat, res_labels_to_index = get_adjmat(res_att_mat)
            res_G = nx.from_numpy_array(res_adj_mat, create_using=nx.DiGraph())

            # Assign edge capacities only for existing edges
            for i, j in res_G.edges():
                res_G[i][j]['capacity'] = res_adj_mat[i, j]

            input_nodes = list(res_labels_to_index.keys())  # All nodes are input nodes
            output_nodes = input_nodes[:]  # All nodes are output nodes

            # Compute flow relevance for the current batch
            flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes, output_nodes, seq_len)

            # Store the results in the tensor
            all_layers_flow_relevance[layer, batch_idx] = flow_values

    return all_layers_flow_relevance # [num_layers, batch_size, seq_len, seq_len]

subset = 2
if __name__ == "__main__":

    model_type = "causal" #'qa' for task3
    task = "task2"
    model, tokenizer = load_llama(model_type=model_type)

    # # Load the sentences
    # with open('materials/sentences.json', 'r') as f:
    #     sentences = json.load(f)
    
    # # Subset for testing
    # if subset:
    #     sentences = sentences[:subset]

    # encodings, word_mappings, prompt_len = encode_input(sentences, tokenizer, task)

    # attention = get_attention(model, encodings) # [num_layers, batch_size, num_heads, seq_len, seq_len]

    # torch.save({
    #     'attention': attention,
    #     'word_mappings': word_mappings,
    #     'prompt_len': prompt_len,
    #     'input_ids': encodings['input_ids'],
    # }, f"/scratch/7982399/thesis/outputs/{task}/attention_data.pt")

    # # Load the attention data
    # loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/attention_data.pt")

    # # Extract raw attention over the input
    # attention_tensor = loaded_data['attention']  # [num_layers, batch_size, num_heads, seq_len, seq_len]
    # input_ids = loaded_data['input_ids']  # [batch_size, seq_len]

    # print("Computing flow relevance scores...")
    
    # # Compute flow relevance for all layers and batches
    # all_examples_flow_relevance = compute_flow_relevance(attention_tensor)

    # # Save the results
    # np.save(f'outputs/{task}/attention_flow.npy', all_examples_flow_relevance)

    # Load the saved attention flow data
    attention_flow = np.load('outputs/task2/attention_flow.npy')

    # Print the type and shape of the loaded data
    print(attention_flow.shape)