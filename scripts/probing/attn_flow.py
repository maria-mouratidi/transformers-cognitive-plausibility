import networkx as nx
import torch
import multiprocessing as mp
import os
from typing import List, Tuple, Dict

# Adjusted to work with torch tensors
def get_adjacency_matrix(mat: torch.Tensor, input_tokens) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Constructs an adjacency matrix as a torch tensor and maps token labels to indices.
    mat is expected to be of shape [n_layers, seq_len, seq_len].
    """
    n_layers, length, _ = mat.shape
    adj_mat = torch.zeros(((n_layers+1) * length, (n_layers+1) * length), device=mat.device)
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
        return torch.zeros(seq_len, dtype=torch.float32, device=attention_tensor.device)
    
    res_att_mat = attention_tensor.mean(dim=1)  # Average across heads new shape: [num_layers, seq_len, seq_len]
    identity_matrix = torch.eye(res_att_mat.shape[1], device=attention_tensor.device)[None,...]  # [1, seq_len, seq_len]
    res_att_mat += identity_matrix # add identity matrices
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
    print("initial flow shape:", attention.shape)
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
                #print("word_idx", word_idx, "word:", word, "prev tokens:", prev_tokens, "token_idx:", token_idx)
                #print(token_attention)
                token_attentions.append(token_attention)

            token_attentions = torch.stack(token_attentions)  # [seq_len, num_tokens]
            if reduction == "mean":
                word_attention = token_attentions.mean(dim=0)  # [seq_len]
            elif reduction == "max":
                word_attention = token_attentions.max(dim=0)[0]
            else:
                raise ValueError("Reduction method must be either 'mean' or 'max'")
            mean_incoming_attention = word_attention[:].mean(-1)
            word_attentions[sentence_idx, word_idx, :] = word_attention
    
    word_attentions = word_attentions[:word_idx].mean(dim=-1)  # [batch_size, max_words]
    # Remove prompt words
    return word_attentions[:, prompt_len:]  # [batch_size, max_words - prompt_len]



if __name__ == "__main__":
    task = "task2"

    checkpoint_file = f"/scratch/7982399/thesis/outputs/{task}/flow/attention_processed.pt"
    flow = torch.load(checkpoint_file)  # Check if the file exists
    # print(torch.any(flow != 0, dim=(1, 2)).nonzero(as_tuple=True)[0])
    # print(torch.all(flow[45] == 0, dim=0).nonzero(as_tuple=True)[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt", map_location=device)
    attention_tensor = loaded_data['attention'].to(device)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
    input_ids = loaded_data['input_ids'].to(device)         # [batch_size, seq_len]
    word_mappings = loaded_data['word_mappings']           # List of token counts for each word in each sentence
    prompt_len = loaded_data['prompt_len']                 # Length of prompt
    # _, batch_size, _, seq_len, _ = attention_tensor.shape
    flow_processed = process_attention_flow(flow, word_mappings, prompt_len, reduction="max")
    print(flow_processed[0])
    # # Load or init checkpoint
    # if os.path.exists(checkpoint_file):
    #     flow_results = torch.load(checkpoint_file, map_location=device)
    # else:
    #     flow_results = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=device)

    # # Save utility
    # def save_checkpoint(tensor, path):
    #     temp_path = path + ".tmp"
    #     torch.save(tensor, temp_path)
    #     os.replace(temp_path, path)

    # # Compute one output index

    # def compute_output(args):
    #     attention_np, token_labels, output_idx = args
    #     attention_tensor_cpu = torch.from_numpy(attention_np)
    #     return output_idx, get_flow_relevance(attention_tensor_cpu, token_labels, layer=31, output_index=output_idx)

    # # Cap CPU usage slightly below max to avoid oversubscription
    # MAX_WORKERS = min(30, mp.cpu_count())

    # # Main loop: batch-wise
    # for batch in range(43, batch_size): #13 and 25 batches may be incomplete

    #     batch_attention_tensor = attention_tensor[:, batch].cpu()  # [num_layers, num_heads, seq_len, seq_len]
    #     input_ids_batch = input_ids[batch]
    #     token_labels = [str(token.item()) for token in input_ids_batch]

    #     # Find which output indices still need processing
    #     remaining_outputs = [
    #         (batch_attention_tensor.numpy(), token_labels, output_idx)
    #         for output_idx in range(seq_len)
    #         if torch.all(flow_results[batch, output_idx] == 0)
    #     ]

    #     if not remaining_outputs:
    #         continue

    #     # Parallel processing
    #     with mp.Pool(processes=MAX_WORKERS) as pool:
    #         results = pool.map(compute_output, remaining_outputs)

    #     for output_idx, flow in results:
    #         flow_results[batch, output_idx] = flow
    #         save_checkpoint(flow_results, checkpoint_file)