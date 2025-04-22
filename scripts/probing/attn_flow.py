import networkx as nx
import numpy as np
import torch
from typing import List, Tuple, Dict
import multiprocessing as mp
from tqdm import tqdm
from scripts.probing.load_model import *
from scripts.probing.raw import *

# Adjusted from https://github.com/oeberle/task_gaze_transformers

def get_adjacency_matrix(attention_matrix: np.ndarray, input_tokens: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Constructs an adjacency matrix from the attention matrix and maps token labels to indices.

    Args:
        attention_matrix: attention weights [n_layers, seq_len, seq_len]
        input_tokens: list of sequence tokens

    Returns:
        adj_matrix:  adjacency matrix for the attention flow graph.
        labels_to_index: a map of token labels to their indices in the adjacency matrix.
    """
    n_layers, seq_len, _ = attention_matrix.shape
    adj_matrix = np.zeros(((n_layers + 1) * seq_len, (n_layers + 1) * seq_len))
    labels_to_index = {}

    # Map input tokens to their indices
    for token_idx in np.arange(seq_len):
        labels_to_index[f"{token_idx}_{input_tokens[token_idx]}"] = token_idx

    # Build adjacency matrix layer by layer
    for layer in np.arange(1, n_layers + 1):
        for from_idx in np.arange(seq_len):
            src_idx = layer * seq_len + from_idx
            src_label = f"L{layer}_{from_idx}"
            labels_to_index[src_label] = src_idx

            for to_idx in np.arange(seq_len):
                target_idx = (layer - 1) * seq_len + to_idx
                adj_matrix[src_idx][target_idx] = attention_matrix[layer - 1][from_idx][to_idx]

    return adj_matrix, labels_to_index

def compute_node_flow(G: nx.DiGraph, labels_to_index: Dict[str, int], input_nodes: List[str], 
                           output_nodes: List[str], seq_len: int) -> np.ndarray:
    """
    Computes the attention flow values between input and output tokens in a directed graph.

    Args:
        G: directed graph representing attention flow.
        labels_to_index: a map of token labels to their indices in the graph.
        input_tokens: the input token labels (source nodes).
        output_tokens: the output token labels (target nodes).
        seq_len: length of the input sequence.

    Returns:
        flow_values: the flow values between nodes in the graph.
    """
    num_nodes = len(labels_to_index)
    flow_values = np.zeros((num_nodes, num_nodes))

    # Iterate over all output tokens
    for key in output_nodes:
        if key not in input_nodes:
            curr_layer = int(labels_to_index[key] / seq_len)  
            pre_layer = curr_layer - 1  
            u = labels_to_index[key]  # Get the index of the output token

            # Compute flow values for each input token
            for in_key in input_nodes:
                v = labels_to_index[in_key] 
                flow_value = nx.maximum_flow_value(G, u, v)  # Compute max flow
                flow_values[u][pre_layer * seq_len + v] = flow_value

            # Normalize flow values for the current output token
            flow_values[u] /= flow_values[u].sum()

    return flow_values

def compute_flow_relevance(attention_tensor: np.ndarray, input_tokens: List[str], layer: int) -> np.ndarray:
    
    res_att_mat = attention_tensor.sum(axis=1) / attention_tensor.shape[1] # Average attention across heads
    res_att_mat += np.eye(res_att_mat.shape[1])[None, ...] # Add self-attention
    res_att_mat /= res_att_mat.sum(axis=-1)[..., None] # Normalize?

    A, labels_to_index = get_adjacency_matrix(res_att_mat, input_tokens)
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph()) # Convert adjacency matrix to a directed graph
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')

    input_nodes, output_nodes = [], []
    for key in labels_to_index:
        if key.startswith('L'+str(layer+1)+'_'): # Next layer output tokens
            output_nodes.append(key)
        if labels_to_index[key] < attention_tensor.shape[-1]: # Input tokens for the current layer
            input_nodes.append(key)
    
    # Compute flow values between input and output tokens
    flow_values = compute_node_flow(G, labels_to_index, input_nodes, output_nodes, length=attention_matrix.shape[-1])
    
    n_layers, length = attention_tensor.shape[0], attention_tensor.shape[-1]
    final_layer_flow = flow_values[(layer+1)*length: (layer+2)*length,layer*length: (layer+1)*length]
    flow_relevance = final_layer_flow.sum(axis=0)

    return flow_relevance

def compute_flow_relevance_for_all_layers(encoded: np.ndarray, attention_tensor: np.ndarray, 
                                          token_labels: List[str], layers: List[int]) -> List[np.ndarray]:
    """
    Computes the flow relevance for all specified layers, handling padding and token cropping.

    Args:
        encoded_tokens: A NumPy array representing the encoded input tokens.
        attention_tensor: A NumPy array of shape [batch_size, num_heads, seq_len, seq_len] representing attention weights.
        token_labels: A list of token labels corresponding to the input sequence.
        layers: A list of layer indices for which to compute flow relevance.

    Returns:
        A list of NumPy arrays, each representing the flow relevance for a specific layer.
    """
    num_layers, batch_size, num_heads, seq_len, _ = attention_tensor.shape
    assert batch_size == 1
    attn_cropped = attention_tensor[:, 0, :, :seq_len, :seq_len]
    tokens_cropped = token_labels[:seq_len]
    all_layers_flow_relevance=[]
    for l in layers:
        attention_relevance = compute_flow_relevance(attn_cropped, tokens_cropped, layer=l)
        all_layers_flow_relevance.append(attention_relevance)
        
    return all_layers_flow_relevance

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