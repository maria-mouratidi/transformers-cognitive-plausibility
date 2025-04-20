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

            # Aggregate attention across heads
            res_att_mat = batch_attention.sum(axis=0) / num_heads

            # Apply causal masking (upper triangular mask)
            causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
            res_att_mat *= causal_mask

            # Add self-loop and normalize
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

    return all_layers_flow_relevance  # [num_layers, batch_size, seq_len, seq_len]

import numpy as np

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
    # }, f"/scratch/7982399/thesis/outputs/{task}/attention_data.pt")

    # Load the attention data
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/attention_data.pt")

    # Extract raw attention over the input
    attention_tensor = loaded_data['attention']  # [num_layers, batch_size, num_heads, seq_len, seq_len]
    input_ids = loaded_data['input_ids']  # [batch_size, seq_len]
    word_mappings = loaded_data['word_mappings']  # List of word mappings
    prompt_len = loaded_data['prompt_len']  # Length of the prompt

    # print("Raw attention shape: ", attention_tensor.shape)
    
    # # Compute flow relevance for all layers and batches
    # print("Computing flow relevance scores...")
    # all_examples_flow_relevance = compute_flow_relevance(attention_tensor)

    # # # Save the results
    # np.save(f'outputs/{task}/attention_flow.npy', all_examples_flow_relevance)

    # Load the saved attention flow data
    attention_flow = np.load('outputs/task2/attention_processed.npy')

    # Print the type and shape of the loaded data
    print(attention_flow.shape)

    # Process the attention flow data
    attention_processed = process_attention_flow(attention_flow, word_mappings, prompt_len)
    print("Processed attention shape: ", attention_processed.shape)


    #TODO: refactor to use torch instead of numpy. or the opposite for raw attn