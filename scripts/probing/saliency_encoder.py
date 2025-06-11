import torch
import numpy as np
import gc
from scripts.probing.load_model import load_bert
from scripts.probing.raw_encoder import encode_input, get_word_mappings
import pickle
import json
from tqdm import tqdm

def register_embedding_list_hook(model, embedding_list):
    def forward_hook(module, inputs, output):
        embedding_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.bert.embeddings.word_embeddings
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embedding_gradients):
    def hook_layers(module, grad_in, grad_out):
        embedding_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.get_input_embeddings()
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook

def lm_saliency_all_tokens(model, input_ids, input_mask):
    torch.enable_grad()
    device = model.device

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    all_token_grads = []
    all_token_embeds = []

    # Iterate over each token position in the input sequence
    for pos in range(1, input_ids.shape[0]):  # skip position 0 (no previous tokens)
        embeddings_list = []
        gradients_list = []

        handle = register_embedding_list_hook(model, embeddings_list)
        hook = register_embedding_gradient_hooks(model, gradients_list)

        model.zero_grad()

        outputs = model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
        logits = outputs.logits # [batch_size, seq_len, vocab_size]

        target_token_id = input_ids[pos]
        token_logit = logits[0, pos, target_token_id] # Influence of each input token in predicting pos
        token_logit.backward()

        handle.remove()
        hook.remove()

        grads = np.array(gradients_list).squeeze()  # [seq_len, hidden_dim]
        embeds = np.array(embeddings_list).squeeze()

        all_token_grads.append(grads.copy())
        all_token_embeds.append(embeds.copy())

        del gradients_list, embeddings_list
        gc.collect()

    return np.array(all_token_grads), np.array(all_token_embeds)


def merge_word_gradients(single_word_mapping, gradients):
    word_gradients = []

    for _, num_tokens, _ in single_word_mapping:
        word_gradient = gradients[:num_tokens].sum(axis=0)  # Aggregate gradients for subtokens
        word_gradients.append(word_gradient)
        gradients = gradients[num_tokens:]  # Move to the next set of subtokens

    return np.array(word_gradients).squeeze()

def l1_grad_norm(grads, word_mapping, normalize=False):
    """ Calculate the L1 norm of gradients (single batch only)"""

    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()
    l1_grad = merge_word_gradients(word_mapping, l1_grad)
    
    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    
    return l1_grad

if __name__ == "__main__":
    model, tokenizer = load_bert(maskedLM=True)
    model.eval()
    device = model.device
    task = "none"

    sentences = ["The quick fox.", "Did not see the fox."]
    sentences = [sentence.split() for sentence in sentences]
    sentences = [["The", "quick", "fox", "."], ["Did", "not", "see", "the", "fox", "."]]
    # with open(f'materials/sentences_{task}.json', 'r') as f:
    #     sentences = json.load(f)

    # Batch encode all sentences and get word mappings
    inputs, sentences, prompt_len = encode_input(sentences, tokenizer, task=task)
    word_mappings = get_word_mappings(sentences, inputs, tokenizer)
    
    saliency = []
    # Compute saliency for each sentence separately
    for sentence_idx in tqdm(range(len(sentences)), total=len(sentences), desc="Processing sentences"):

        # Extract inputs and word mappings for the current sentence
        input_ids = inputs["input_ids"][sentence_idx]
        input_mask = inputs["attention_mask"][sentence_idx]
        sentence_word_mapping = word_mappings[sentence_idx]

        # Clear gradients before processing
        model.zero_grad()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compute saliency for this sentence
        grads_all, embeds_all = lm_saliency_all_tokens(model, input_ids, input_mask)

        # sentence_saliency = np.zeros((len(grads_all), len(sentence_word_mapping)), dtype=np.float32)
        # for t, grads in enumerate(grads_all):
        #     l1_grad = l1_grad_norm(grads, sentence_word_mapping, normalize=True)
        #     sentence_saliency[t] = l1_grad

        sentence_saliency = grads_all.copy()  # [n_tokens, n_hidden]
        
        avg_saliency = np.mean((sentence_saliency), axis=0) # New shape: n_words
        #                                         ^^^^^ axis 0 = across output positions

        saliency.append(avg_saliency)

        # Clean up after processing
        del grads_all, embeds_all
        gc.collect()

    print(saliency[:2])

    # with open(f"outputs/saliency/{task}/bert/saliency_data.pkl", "wb") as f:
    #     pickle.dump(saliency, f)


# Gradient:
# How much does changing the input embedding of token at position i
# affect the prediction (logit) of the token at position pos?

# So saliency_tensor[t, i] tells you:
# Saliency of predicting token at position t w.r.t. embedding at position i
