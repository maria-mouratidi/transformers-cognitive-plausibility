import networkx as nx
import torch
import os
from typing import List, Tuple, Dict

# Adjusted to work with torch tensors
def get_adjacency_matrix(mat: torch.Tensor, input_tokens) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Constructs an adjacency matrix as a torch tensor and maps token labels to indices.
    mat is expected to be of shape [n_layers, seq_len, seq_len].
    """
    n_layers, length, _ = mat.shape
    adj_mat = torch.zeros(((n_layers+1) * length, (n_layers+1) * length))
    labels_to_index = {}
    
    for token_idx in range(length):
        labels_to_index[str(token_idx) + "_" + input_tokens[token_idx]] = token_idx

    for layer in range(1, n_layers+1):
        for src_token in range(length):
            index_from = (layer) * length + src_token
            label = "L" + str(layer)+ "_" + str(src_token)
            labels_to_index[label] = index_from
            for target_token in range(length):
                index_to = (layer-1) * length + target_token
                adj_mat[index_from][index_to] = mat[layer-1][src_token][target_token]

    return adj_mat, labels_to_index 

def compute_node_flow(G: nx.DiGraph, labels_to_index: Dict[str, int], input_nodes: List[str], 
                      output_nodes: List[str], seq_len: int) -> torch.Tensor:
    """
    Computes the attention flow values between input and output tokens in a directed graph.
    Uses networkx maximum flow on a numpy array representation.
    """
    num_nodes = len(labels_to_index)
    flow_values = torch.zeros((num_nodes, num_nodes))
    
    for key in output_nodes:
        if key not in input_nodes:
            curr_layer = int(labels_to_index[key] / seq_len)
            pre_layer = curr_layer - 1  
            u = labels_to_index[key]
            
            for input_node_key in input_nodes:
                v = labels_to_index[input_node_key]
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer * seq_len + v] = float(flow_value)
                
            # Normalize the current output token's flow values
            flow_values[u] /= flow_values[u].sum() 

    return flow_values

def get_flow_relevance(attention_tensor: torch.Tensor, input_tokens: List[str], layer: int, output_index, pad_id = '128001') -> torch.Tensor:
    """
    Computes the flow relevance for a specified layer.
    Expects attention_tensor to have shape [num_layers, num_heads, seq_len, seq_len].
    """
    seq_len = attention_tensor.shape[-1]

    if input_tokens[output_index] == pad_id: 
        return torch.zeros(seq_len, dtype=torch.float32)
 
    res_att_mat = attention_tensor.mean(dim=1)  # Average across heads new shape: [num_layers, seq_len, seq_len]
    res_att_mat = res_att_mat + torch.eye(res_att_mat.shape[1])[None,...] # add identify matrices
    res_att_mat = res_att_mat / res_att_mat.sum(dim=-1)[..., None]
        
    # Build adjacency matrix (using torch) and create a graph (networkx requires numpy)
    A, labels_to_index = get_adjacency_matrix(res_att_mat, input_tokens)
    A_np = A.cpu().numpy()
    G = nx.from_numpy_array(A_np, create_using=nx.DiGraph())
    capacities = {(i, j): A_np[i, j] for i in range(A_np.shape[0]) for j in range(A_np.shape[1])}
    nx.set_edge_attributes(G, capacities, 'capacity')
    
    # Identify nodes for input (layer 0) and output (layer layer+1)
    input_nodes = []
    output_nodes = ['L'+str(layer+1)+'_'+str(output_index)]

    for key in labels_to_index:
        if labels_to_index[key] < seq_len:
            input_nodes.append(key)
    
    # Compute flow values and convert to torch tensor
    flow_values = compute_node_flow(G, labels_to_index, input_nodes, output_nodes, seq_len)
    final_layer_flow = flow_values[(layer + 1)*seq_len:, layer*seq_len:(layer + 1)*seq_len]

    return final_layer_flow[output_index]

def process_attention_flow(attention: torch.Tensor, word_mappings: List[List[Tuple[str, int]]],
                           prompt_len: int, reduction: str = "mean") -> torch.Tensor:
    """
    Extract word-level attention from token-level attention weights for attention flow.
    Expects attention tensor of shape [num_layers, batch_size, seq_len, seq_len].
    """
    print("initial attention shape:", attention.shape)
    batch_size, seq_len, _ = attention.shape
    max_words = max(len(word_map) for word_map in word_mappings)
    word_attentions = torch.zeros((batch_size, max_words, seq_len), dtype=torch.float32)
    
    for sentence_idx, word_map in enumerate(word_mappings):
        for word_idx, (word, num_tokens) in enumerate(word_map):
            token_attentions = []
            for n_token in range(num_tokens):
                prev_tokens = word_map[:word_idx]
                token_idx = sum(token[1] for token in prev_tokens) + n_token
                token_attention = attention[sentence_idx, token_idx, :] #now we index dim 1 instead of 2 like in raw attention because flow is aggregated differently
                token_attentions.append(token_attention)
                print(token_attention)

            token_attentions = torch.stack(token_attentions)  # [seq_len, num_tokens]
            if reduction == "mean":
                word_attention = token_attentions.mean(dim=0)  # [seq_len]
            elif reduction == "max":
                word_attention = token_attentions.max(dim=0)[0]
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")
        if sentence_idx == 0:
            break

            word_attentions[sentence_idx, word_idx, :] = word_attention
    word_attentions = word_attentions.mean(dim=-1)  # [batch_size, max_words]
    # Remove prompt words
    return word_attentions[:, prompt_len:]  # [batch_size, max_words - prompt_len]

if __name__ == "__main__":
    task = "task2"
    # # # Load the attention data (assumed already saved as torch tensors)
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")
    # attention_tensor = loaded_data['attention']            # [num_layers, batch_size, num_heads, seq_len, seq_len]
    word_mappings = loaded_data['word_mappings']  # List of word mappings for each sentence
    prompt_len = loaded_data['prompt_len']  # Length of the prompt
    # input_ids = loaded_data['input_ids']                  # [batch_size, seq_len]

    # _, batch_size, _, seq_len, _ = attention_tensor.shape
    # flow_results = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32)
    # checkpoint_file = f"/scratch/7982399/thesis/outputs/{task}/flow/attention_flow_checkpoint.pt"

    # for batch in range(batch_size):
    #     if os.path.exists(checkpoint_file):
    #         # Optionally load previously computed progress here
    #         flow_results = torch.load(checkpoint_file)
    #     else:
    #         print("Checkpoint file not found, starting from scratch.")

    #     attention_tensor = attention_tensor[:, batch, ...]      # Now shape: [num_layers, num_heads, seq_len, seq_len]
    #     input_ids_batch = input_ids[batch]                            # Now shape: [seq_len]
    #     token_labels = [str(token.item()) for token in input_ids_batch]

    #     for output_idx in range(seq_len):
    #         # Compute flow relevance for the last layer
    #         flow = get_flow_relevance(attention_tensor, token_labels, layer=31, output_index=output_idx)
    #         flow_results[batch, output_idx, :] = flow
    #         torch.save(flow_results, checkpoint_file)  # Save progress

    flow = torch.load(f"/scratch/7982399/thesis/outputs/{task}/flow/attention_flow_checkpoint.pt")
    # print(flow.shape)
    # print(flow[0])
    flow_processed = process_attention_flow(flow, word_mappings, prompt_len, reduction="max")
    # print(flow_processed.shape)
    # torch.save(flow_processed, f"/scratch/7982399/thesis/outputs/{task}/flow/attention_flow_processed.pt")
    #flow_processed = torch.load(f"/scratch/7982399/thesis/outputs/{task}/flow/attention_flow_processed.pt")
    # print(flow_processed.shape)
    # print(flow_processed[0])