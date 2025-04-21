import networkx as nx
import numpy as np
import json
import torch
from typing import List, Tuple
from scripts.probing.load_model import *
from scripts.probing.raw import *

def get_adjacency_mat(attention: np.ndarray, input_tokens: List[str] = None) -> Tuple[np.ndarray, dict]:
    """
    Build adjacency matrix across all layers.

    Args:
        attention: Array of shape [num_layers, seq_len, seq_len]
        input_tokens: Optional, just for reference. Not used here.

    Returns:
        adj_mat: Adjacency matrix of shape [num_nodes, num_nodes]
        labels_to_index: Dict mapping node labels like "L1_3" to graph indices.
    """
    num_layers, seq_len, _ = attention.shape
    num_nodes = (num_layers + 1) * seq_len
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    labels_to_index = {}

    # Assign index to each node L0_i, L1_i, ..., L_N_i
    for layer in range(num_layers + 1):
        for i in range(seq_len):
            label = f"L{layer}_{i}"
            idx = layer * seq_len + i
            labels_to_index[label] = idx

    # Fill adjacency matrix: layer L → layer L+1 using multi_layer_att[L]
    for layer in range(num_layers):
        for i in range(seq_len):
            for j in range(seq_len):
                src = labels_to_index[f"L{layer}_{i}"]
                tgt = labels_to_index[f"L{layer+1}_{j}"]
                if j <= i:  # Causal attention
                    adj_mat[src, tgt] = attention[layer, i, j]

    return adj_mat, labels_to_index


def compute_single_node_flow(G, labels_to_index, input_nodes, output_node, seq_len):
    u = labels_to_index[output_node]
    layer = int(output_node.split("_")[0][1:]) - 1
    flow_row = np.zeros(seq_len, dtype=np.float32)

    for inp_node in input_nodes:
        v = labels_to_index[inp_node]
        try:
            flow = nx.maximum_flow_value(G, v, u)
        except:
            flow = 0.0
        flow_row[v] = flow

    total = flow_row.sum()
    if total > 0:
        flow_row /= total

    return flow_row

def compute_flow_relevance(raw_attention: np.ndarray, input_ids: List[List[str]]) -> np.ndarray:
    """
    Computes flow relevance for all layers, batches, and token pairs.

    Args:
        raw_attention: Attention tensor [num_layers, batch_size, num_heads, seq_len, seq_len]
        input_token_list: List of tokenized input strings, one list of tokens per batch item.

    Returns:
        np.ndarray of shape [num_layers, batch_size, seq_len, seq_len]
    """
    num_layers, batch_size, num_heads, seq_len, _ = raw_attention.shape
    flow_relevance = np.zeros((num_layers, batch_size, seq_len, seq_len), dtype=np.float32)
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32)) 
    identity_attention = np.eye(seq_len, dtype=np.float32)  
    input_nodes = [f"L0_{i}" for i in range(seq_len)]
    
    for b in range(batch_size):
        tokens = input_ids[b]  # [seq_len]
        attention = raw_attention[:, b]  # [num_layers, num_heads, seq_len, seq_len]

        # Step 1: Average heads and normalize
        avg_attention = attention.mean(axis=1)  # [num_layers, seq_len, seq_len]
        avg_attention = avg_attention * causal_mask[np.newaxis, :, :]  # Apply causal mask
        avg_attention = avg_attention + identity_attention  # Apply residual connection
        token_attention = avg_attention.sum(axis=-1, keepdims=True)
        avg_attention = avg_attention / token_attention

        # Step 2: Build graph across layers
        adjacency_mat, labels_to_index = get_adjacency_mat(avg_attention, input_tokens=tokens)
        G = nx.from_numpy_array(adjacency_mat, create_using=nx.DiGraph())
        for i, j in G.edges():
            G[i][j]["capacity"] = adjacency_mat[i, j]

        # Step 3: Compute flow relevance for each layer and output token
        for layer in range(num_layers):
            for j in range(seq_len):
                output_node = f"L{layer + 1}_{j}"
                relevance_row = compute_single_node_flow(G, labels_to_index, input_nodes, output_node, seq_len)
                flow_relevance[layer, b, :, j] = relevance_row

    return flow_relevance # [num_layers, batch_size, seq_len, seq_len]


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
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

    # Extract raw attention over the input
    attention_tensor = loaded_data['attention']  # [num_layers, batch_size, num_heads, seq_len, seq_len]
    input_ids = loaded_data['input_ids']  # [batch_size, seq_len]
    word_mappings = loaded_data['word_mappings']  # List of word mappings
    prompt_len = loaded_data['prompt_len']  # Length of the prompt

    print("Raw attention shape: ", attention_tensor.shape)
    
    # Compute flow relevance for all layers and batches
    print("Computing flow relevance scores...")

    all_examples_flow_relevance = compute_flow_relevance(attention_tensor, input_ids)
    #print("Flow relevance: ", all_examples_flow_relevance[0,0,0])

    # Save the results
    np.save(f'outputs/{task}/flow/attention_flow.npy', all_examples_flow_relevance)

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