import torch
import numpy as np
from scripts.probing.load_model import load_llama

def compute_sensitivity(model, tokenizer, token_ids):
    """
    Compute saliency (gradient-based sensitivity) of each token in a LLaMA model.
    Optimized for GPU usage and FP16 inference.
    """
    model.eval()
    device = next(model.parameters()).device
    sensitivity_data = []

    for masked_token_index in range(len(token_ids)):
        # Create input tensor and move to GPU
        token_id_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)  # (1, seq_len)

        # Get embeddings and enable gradient tracking only on them
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(token_id_tensor).detach()
        inputs_embeds.requires_grad_(True)

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Extract the logit corresponding to the current token (target self-prediction)
        target_token_id = token_ids[masked_token_index]
        target_logit = logits[0, masked_token_index, target_token_id]

        # Backward pass
        model.zero_grad()
        target_logit.backward()

        # Compute L2 norm of gradients across embedding dimension
        grads = inputs_embeds.grad  # (1, seq_len, hidden_dim)
        grads_norm = torch.norm(grads, dim=2)  # (1, seq_len)
        grads_norm = grads_norm / grads_norm.max()  # Normalize for stability

        # Store sensitivity
        token = tokenizer.convert_ids_to_tokens(token_ids[masked_token_index])
        sensitivity = grads_norm[0].detach().cpu().tolist()
        sensitivity_data.append({'token': token, 'sensitivity': sensitivity})

    return sensitivity_data

def extract_relative_saliency(model, tokenizer, sentence):
    """
    Compute relative saliency for a given sentence using LLaMA model.
    Returns tokens and their overall importance.
    """
    # Tokenize without adding special tokens
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    sensitivity_data = compute_sensitivity(model, tokenizer, token_ids)

    distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
    tokens = [entry["token"] for entry in sensitivity_data]
    saliency = np.sum(distributed_sensitivity, axis=0)  # Aggregate across tokens

    return tokens, saliency

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_llama("causal")
    # Example sentence
    sentence = "The quick fox."

    # Compute and print saliency
    tokens, saliency = extract_relative_saliency(model, tokenizer, sentence)
    for tok, score in zip(tokens, saliency):
        print(f"{tok:>12s}: {score:.4f}")
