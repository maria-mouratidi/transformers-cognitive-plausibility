import torch
import numpy as np
from scripts.probing.load_model import load_llama
from scripts.probing.raw import encode_input, get_word_mappings

def register_hooks(model):
    """
    Hooks to capture embeddings and gradients during forward and backward passes.

    Returns:
        A tuple (embeddings_list, gradients_list, remove_hooks) where:
        - embeddings_list: List to store embeddings.
        - gradients_list: List to store gradients.
        - remove_hooks: Function to remove the registered hooks.
    """
    embeddings_list = []
    gradients_list = []

    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())

    def backward_hook(module, grad_in, grad_out):
        gradients_list.append(grad_out[0].detach().cpu().numpy())

    embedding_layer = model.get_input_embeddings()
    forward_handle = embedding_layer.register_forward_hook(forward_hook)
    backward_handle = embedding_layer.register_full_backward_hook(backward_hook)

    def remove_hooks():
        forward_handle.remove()
        backward_handle.remove()

    return embeddings_list, gradients_list, remove_hooks

def lm_saliency(model, input_ids, input_mask, label_id):
    
    model.zero_grad()  # Clear previous gradients
    torch.enable_grad() # Enable gradient computation
    model.eval() # Set model to evaluation mode
    device = model.device

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    
    embeddings_list, gradients_list, remove_hooks = register_hooks(model)

    outputs = model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0)) # Forward pass
    outputs.logits[0][-1][label_id].backward() # Compute gradients w.r.t. the last token

    remove_hooks() # Remove hooks after capturing gradients

    gradients_list = np.array(gradients_list).squeeze() 
    embeddings_list = np.array(embeddings_list).squeeze()

    return gradients_list, embeddings_list


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
    device = "cpu"
    task = "none"

    sentences = ["The quick fox.", "Did not see that coming."]
    #sentences = ["The quick fox."]
    sentences = [sentence.split() for sentence in sentences]
    inputs, sentences_full, prompt_len = encode_input(sentences, tokenizer, task=task)

    input_ids = inputs["input_ids"].to(device).tolist()
    input_mask = inputs["attention_mask"].to(device).tolist()
    word_mappings = get_word_mappings(sentences_full, inputs, tokenizer)

    for sentence_idx, sentence in enumerate(sentences):

        input_id = torch.tensor(input_ids[sentence_idx]).to(device).unsqueeze(0)
        label_id = model(input_ids=input_id).logits[0, -1].argmax().item()

        # Assuming you want to inspect the most probable next word
        grads, embeds = lm_saliency(
            model,
            input_ids[sentence_idx],
            input_mask[sentence_idx],
            label_id=label_id
        )

        l1_grad = l1_grad_norm(grads, word_mappings[sentence_idx], normalize=True)
        print(f"Sentence: {' '.join(sentence)}")
        print("L1 Gradient Norm:", l1_grad)