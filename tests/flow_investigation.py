import torch
from pprint import pprint
from scipy.sparse import csr_matrix, csr_array
from scipy.sparse.csgraph import maximum_flow
from torch.nn import functional as F


attn = torch.rand((3, 1, 1, 3, 3)) ## layer, batch, head, length, length
attn = torch.randn(32, 300, 32, 43,43)
attn = F.softmax(attn, dim=-1)
def create_graph(attn, batch):
    n_layers, n_batches, n_heads, length, _ = attn.shape
    attn = attn.mean(dim=2) # new shape: (n_layers, n_batches, length, length)
    length = 43

    # Initialize the adjacency matrix (+1 for supersource and supersink)
    adj = torch.zeros(((n_layers+1) * length + 2, (n_layers+1) * length + 2)) #rows = source per layer, cols = sink per layer
    inf_cap = int(10e6) # Supersource and supersink have infinite capacity

    # Source node
    adj[0][1: 1+length] = inf_cap

    # Sink node
    #adj[-length-1: -1][-1] = inf_cap
    for i in range(length):
        adj[-i-2][-1] = inf_cap


    # Fill with attention values (except first and last which are source and sink)
    for layer in range(n_layers):
        for i in range(length):
            for j in range(length):
                idx_i = 1+length*layer + i
                idx_j = 1+length*(layer+1) + j
                attn_value = int(attn[layer, batch, i, j] * 1e6) # Max flow is for ints
                adj[idx_i][idx_j] = attn_value
                if layer ==0:
                    print(f"attn_value: {attn_value}, adj[{idx_i}][{idx_j}]: {adj[idx_i][idx_j]}")
                
    #print(adj)
              
    source = 0
    target = length * (n_layers + 1) 

    graph = csr_array(adj, dtype=int)
    print(graph.shape, graph)
    max_flow = maximum_flow(graph, source, target, method='edmonds_karp').flow.toarray()

    return max_flow


loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/task2/raw/attention_data.pt")
attn = loaded_data['attention'] # [num_layers, batch_size, num_heads, seq_len, seq_len]
#mappings = loaded_data['word_mappings'] # [num_layers, batch_size, num_heads, seq_len, seq_len]
#print(attn[0, 0, 0]==0, dim=0)
# print(torch.all(attn[0,0,0] == 0, dim=0).nonzero(as_tuple=True)[0])
# sent1 = attn[0, 0, 0]
# sent2 = attn[0, 1, 0]
# print(attn[0,0,0, 43, 43:])
# torch.set_printoptions(sci_mode=False, linewidth=1000)
torch.set_printoptions(linewidth=1000)
result = create_graph(attn, 0)
print(result)


#FIX: The problem is in populating the adj matrix.
#TODO: redisual connections
#TODO: first token bias
# check whether raw is correct with padding -> pad tokens still attend but this is default behaviro, not an error on my part.