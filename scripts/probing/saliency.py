import torch
import numpy as np
from scripts.probing.load_model import load_llama

def compute_sensitivity(model, tokenizer, sentence):
    """
    Compute saliency (gradient-based sensitivity) of each token in a LLaMA model.
    Optimized for GPU usage and FP16 inference.
    """
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    model.eval()
    device = next(model.parameters()).device
    sensitivity = []

    # Create input tensor and move to GPU
    token_id_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)  # (1, seq_len)
    seq_len = token_id_tensor.shape[1]
    
    # Get embeddings and enable gradient tracking only on them
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(token_id_tensor).detach()
    inputs_embeds.requires_grad_(True)

    # Forward pass
    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Extract the logit corresponding to the current token (target self-prediction)
    for t_prime in range(seq_len):
        model.zero_grad()
        inputs_embeds.grad = None  # Reset gradients
        target_token_id = token_ids[t_prime]
        target_logit = logits[0, t_prime, target_token_id]


        # Backward pass to compute gradients
        target_logit.backward(retain_graph=True)

        # Gradient of output at t' w.r.t. input at t
        grads = inputs_embeds.grad[0]  # (seq_len, hidden_dim)

        # Compute L2 norm of gradients across embedding dimension
        print("\n\n")
        grads_l2 = grads.pow(2).sum(dim=1) # (seq_len,)
        print("L2: ", grads_l2)
        grads_l2_sum = grads_l2.sum()  # Sum across all tokens
        print("L2 sum: ", grads_l2_sum)
        grads_norm = grads_l2_sum / grads_l2.max()  # Normalize for stability
        print("L2 norm: ", grads_norm)

        sensitivity.append(grads_norm.detach().cpu().tolist())
    print(sensitivity)

    return tokens, sensitivity # (seq_len, seq_len)

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_llama("causal")
    # Example sentence
    sentence = "The quick fox."
    tokens, sensitivity = compute_sensitivity(model, tokenizer, sentence)
    for tok, score in zip(tokens, sensitivity):
        print(f"{tok:>12s}: {score}")


    # tokens, saliency = extract_relative_saliency(model, tokenizer, sentence)
    # for tok, score in zip(tokens, saliency):
    #     print(f"{tok:>12s}: {score:.4f}")
