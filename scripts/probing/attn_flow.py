import networkx as nx
import numpy as np
import json
import torch
from typing import List, Tuple
from scripts.probing.load_model import *
from scripts.probing.raw import *

def get_adjmat(attention_matrix: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Converts a single-layer attention matrix into an adjacency matrix for graph processing.
    
    Args:
        attention_matrix: Array of shape [seq_len, seq_len], representing attention for a single layer.
    
    Returns:
        adj_mat: Array of shape [seq_len, seq_len], representing the adjacency matrix.
        labels_to_index: Dictionary mapping token positions to graph node indices.
    """
    seq_len = attention_matrix.shape[0]
    adj_mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    labels_to_index = {}

    for i in range(seq_len):
        for j in range(seq_len):
            # Map token positions to indices
            labels_to_index[i] = i
            labels_to_index[j] = j
            
            # Assign attention weight to adjacency matrix
            adj_mat[i, j] = attention_matrix[i, j]
    
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


def compute_flow_relevance(full_att_mat: np.ndarray, input_tokens: List[str], layer: int) -> np.ndarray:
    """
    Computes flow relevance scores for all words in a sentence.
    
    Args:
        full_att_mat: Attention matrices of shape [num_layers, batch_size, num_heads, seq_len, seq_len]
        input_tokens: List of tokenized words
        layer: Target layer index
    
    Returns:
        A numpy array representing relevance scores for all words in the sentence.
    """
    n_layers, batch_size, num_heads, seq_len, _ = full_att_mat.shape
    all_flow_relevance = []

    # Loop over batch size (each sentence in the batch)
    for batch_idx in range(batch_size):
        # Get the attention matrix for the current batch (averaging over heads)
        batch_att_mat = torch.mean(full_att_mat[:, batch_idx, :, :, :], dim=1).numpy()  # [num_layers, seq_len, seq_len]
        
        # Add identity matrix to self-connect tokens (self-attention)
        res_att_mat = batch_att_mat[layer] + np.eye(seq_len)
        res_att_mat /= np.sum(res_att_mat, axis=-1, keepdims=True)  # Normalize
        
        # Convert attention to adjacency matrix
        res_adj_mat, res_labels_to_index = get_adjmat(res_att_mat)

        # Convert adjacency matrix to networkx graph
        res_G = nx.from_numpy_array(res_adj_mat, create_using=nx.DiGraph())

        # Assign capacity attributes for max-flow computation
        for i in range(res_adj_mat.shape[0]):
            for j in range(res_adj_mat.shape[1]):
                res_G[i][j]['capacity'] = res_adj_mat[i, j]

        # Define input and output nodes
        input_nodes = [key for key in res_labels_to_index if res_labels_to_index[key] < seq_len]
        output_nodes = list(res_labels_to_index.keys())

        # Compute flow relevance
        flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes=input_nodes, output_nodes=output_nodes, seq_len=seq_len)
        all_flow_relevance.append(flow_values)

    return np.array(all_flow_relevance)  # Shape: [batch_size, seq_len, seq_len]

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
    
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/attention_data.pt")

    # Extract  raw attention over the input
    attention_tensor = loaded_data['attention'] #[num_layers, batch_size, num_heads, seq_len, seq_len]
    input_ids = loaded_data['input_ids'] # [batch_size, seq_len]

    print("Computing flow relevance scores...")
    all_examples_flow_relevance = {}
    for l in range(model.config.num_hidden_layers):
        all_examples_flow_relevance[l] = []
        # Get the tokenized sentence
        tokens = tokenizer.decode(input_ids[:][0].numpy())
        
        # Compute flow relevance for each sentence in the batch
        attention_relevance = compute_flow_relevance(attention_tensor, tokens, layer=l)
        
        # Store the results
        all_examples_flow_relevance[l].append(attention_relevance)

    # Save the results
    np.save(f'outputs/{task}/attention_flow', all_examples_flow_relevance)
