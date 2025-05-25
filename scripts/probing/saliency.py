import torch
import numpy as np
from scripts.probing.load_model import load_llama

def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.get_input_embeddings()

    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())

    embedding_layer = model.get_input_embeddings()
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook

def lm_saliency(model, tokenizer, input_ids, input_mask, label_id):

    torch.enable_grad()
    model.eval()
    device = model.device

    embeddings_list = []
    gradients_list = []

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    handle = register_embedding_list_hook(model, embeddings_list)
    hook = register_embedding_gradient_hooks(model, gradients_list)

    model.zero_grad()
    A = model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
    
    A.logits[0][-1][label_id].backward()

    handle.remove()
    hook.remove()

    gradients_list = np.array(gradients_list).squeeze()
    embeddings_list = np.array(embeddings_list).squeeze()
    # Merge tokens into the original words
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return tokens, gradients_list, embeddings_list

def merge_gpt_tokens(tokens, gradients):
    merged_gradients = []
    merged_words = []  # To store the merged words
    word = ""
    word_gradients = 0
    # Merge tokens into the original words
    for i, token in enumerate(tokens):
        if token.startswith("Ġ") or token.startswith("Ċ"):
            if word != "":
                merged_gradients.append(word_gradients)
                merged_words.append(word)  # Append the completed word
                word = ""
                word_gradients = 0
            word = token[1:]
            word_gradients = gradients[i]
        else:
            word += token
            word_gradients += gradients[i]
    if word != "":
        merged_gradients.append(word_gradients)
        merged_words.append(word)  # Append the last word
    return np.array(merged_gradients).squeeze(), merged_words

def l1_grad_norm(tokens, grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()
    l1_grad, merged_tokens = merge_gpt_tokens(tokens, l1_grad)
    
    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    
    return l1_grad, merged_tokens

if __name__ == "__main__":
    model, tokenizer = load_llama("causal")
    device = model.device

    sentence = "The quick fox."
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"][0].to(device).tolist()
    input_mask = inputs["attention_mask"][0].to(device).tolist()

    label_id = model(input_ids=torch.tensor([input_ids]).to(device)).logits[0, -1].argmax().item()

    # Assuming you want to inspect the most probable next word
    tokens, grads, embeds = lm_saliency(
        model,
        tokenizer,
        input_ids,
        input_mask,
        label_id=model(input_ids=torch.tensor([input_ids]).to(device)).logits[0, -1].argmax().item()
    )

    l1_grad, merged_tokens = l1_grad_norm(tokens, grads, normalize=True)
    print("Tokens:", tokens)
    print("L1 Gradient Norm:", l1_grad)
    print("Merged Tokens:", merged_tokens)
