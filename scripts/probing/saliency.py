import torch
import numpy as np
import gc
from scripts.probing.load_model import load_llama
from scripts.probing.raw import encode_input, get_word_mappings

def register_embedding_list_hook(model, embedding_list):
    def forward_hook(module, inputs, output):
        embedding_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.get_input_embeddings()
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embedding_gradients):
    def hook_layers(module, grad_in, grad_out):
        embedding_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.get_input_embeddings()
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook

def lm_saliency(model, input_ids, input_mask, pos=-1):
    
    torch.enable_grad() # Enable gradient computation
    device = model.device

    embeddings_list = []  # To store embeddings
    gradients_list = []  # To store gradients

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    handle = register_embedding_list_hook(model, embeddings_list)
    hook = register_embedding_gradient_hooks(model, gradients_list)

    model.zero_grad()  # Clear previous gradients
    outputs = model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0)) # Forward pass
    logits = outputs.logits

    token_logit = logits[0, pos, input_ids[pos]]
    token_logit.backward()  # Backward pass to compute gradients

    handle.remove()  # Remove the forward hook
    hook.remove()  # Remove the backward hook

    gradients_list = np.array(gradients_list).squeeze() 
    embeddings_list = np.array(embeddings_list).squeeze()

    return gradients_list, embeddings_list

def merge_word_gradients(single_word_mapping, gradients):
    word_gradients = []

    for _, num_tokens, _ in single_word_mapping:
        word_gradient = gradients[:num_tokens].sum(axis=0)  # Aggregate gradients for subtokens
        word_gradients.append(word_gradient)
        gradients = gradients[num_tokens:]  # Move to the next set of subtokens

    return np.array(word_gradients).squeeze()

def l1_grad_norm(grads, single_word_mapping, normalize=False):
    """ Calculate the L1 norm of gradients (single batch only)"""

    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()
    l1_grad = merge_word_gradients(single_word_mapping, l1_grad)
    
    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    
    return l1_grad

if __name__ == "__main__":
    model, tokenizer = load_llama("causal")
    model.eval()
    device = model.device
    task = "task2"

    sentences = ["The quick fox.", "Did not see the fox."]
    #sentences = ["The quick fox."]
    sentences = [sentence.split() for sentence in sentences]


    for sentence_idx, sentence in enumerate(sentences):
        inputs, sentences_full, prompt_len = encode_input([sentence], tokenizer, task=task)
    
        word_mappings = get_word_mappings(sentences_full, inputs, tokenizer)
        input_ids = inputs["input_ids"][0]
        input_mask = inputs["attention_mask"][0]
        print(f"Input IDs: {input_ids}")
        print(f"Input Mask: {input_mask}")

        # Assuming you want to inspect the most probable next word
        grads, embeds = lm_saliency(
            model,
            input_ids,
            input_mask,
        )


        l1_grad = l1_grad_norm(grads, word_mappings[0], normalize=True)
        print(f"Sentence: {' '.join(sentence)}")
        print("L1 Gradient Norm:", l1_grad)
