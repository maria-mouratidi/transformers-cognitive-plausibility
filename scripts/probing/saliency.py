import numpy as np
import tensorflow as tf
import scipy.special

# The code is adapted from https://github.com/felixhultin/cross_lingual_relative_importance/blob/main/extract_model_importance/extract_saliency.py
# The code for calculating sensitivity is based on the TextualHeatmap example by Andreas Madsen: https://colab.research.google.com/github/AndreasMadsen/python-textualheatmap/blob/master/notebooks/huggingface_bert_example.ipynb#scrollTo=X8GJbpoUmYdT
# As described in [Andreas Madsen's distill paper](https://distill.pub/2019/memorization-in-rnns/), the saliency map is computed by measuring the gradient magnitude of the output w.r.t. the input.
def compute_sensitivity(model, tokenizer, embedding_matrix, token_ids):
    vocab_size = embedding_matrix.get_shape()[0]
    sensitivity_data = []

    # Iteratively mask each token in the input
    for masked_token_index in range(len(token_ids)):

        if masked_token_index == 0:
            sensitivity_data.append({'token': '[CLR]', 'sensitivity': [1] + [0] * (len(token_ids) - 1)})

        elif masked_token_index == len(token_ids) - 1:
            sensitivity_data.append({ 'token': '[SEP]', 'sensitivity': [0] * (len(token_ids) - 1) + [1]})

        # Get the actual token
        else:
            target_token = tokenizer.convert_ids_to_tokens(token_ids[masked_token_index])
            # integers are not differentable, so use a one-hot encoding of the intput
            masked_token_ids = token_ids[0:masked_token_index] + [tokenizer.mask_token_id] + token_ids[masked_token_index + 1:]
            token_ids_tensor = torch.tensor([masked_token_ids], dtype=torch.long)
            # Create one-hot encoding of token IDs
            token_ids_tensor_one_hot = torch.nn.functional.one_hot(token_ids_tensor, num_classes=vocab_size).float()
            token_ids_tensor_one_hot.requires_grad = True
            
            
            # To select the correct output, create a masking tensor.
            # tf.gather_nd could also be used, but this is easier.
            output_mask = torch.zeros((1, len(token_ids), vocab_size))
            output_mask[0, masked_token_index, token_ids[masked_token_index]] = 1
            output_mask_tensor = tf.constant(output_mask, dtype='float32')
            # Apply embedding matrix and get model predictions
            inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
            
            # Forward pass
            with torch.no_grad():
                # Some models expect inputs_embeds directly, others as part of a dictionary
                try:
                    outputs = model(inputs_embeds=inputs_embeds)
                except:
                    outputs = model({"inputs_embeds": inputs_embeds})
                    
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Compute gradient for the target token prediction
            predict_mask_correct_token = torch.sum(logits * output_mask)
            
            # Compute gradient
            predict_mask_correct_token.backward()
            
            # Compute sensitivity (L2 norm of gradient)
            sensitivity_non_normalized = torch.norm(token_ids_tensor_one_hot.grad, dim=2)
            
            # Normalize sensitivity
            sensitivity_tensor = sensitivity_non_normalized / torch.max(sensitivity_non_normalized)
            sensitivity = sensitivity_tensor[0].detach().cpu().numpy().tolist()
            
            sensitivity_data.append({'token': target_token, 'sensitivity': sensitivity})

    return sensitivity_data

# We calculate relative saliency by summing the sensitivity a token has with all other tokens
def extract_relative_saliency(model, embeddings,tokenizer, sentence):
    sensitivity_data = compute_sensitivity(model, embeddings, tokenizer, sentence)

    distributed_sensitivity = np.asarray([entry["sensitivity"] for entry in sensitivity_data])
    tokens = [entry["token"] for entry in sensitivity_data]

    # For each token, I sum the sensitivity values it has with all other tokens
    saliency = np.sum(distributed_sensitivity, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    # saliency = scipy.special.softmax(saliency)
    return tokens, saliency


if __name__ == "__main__":
    import torch
    from scripts.probing.load_model import *

    task = "task3" # None, task2, task3
    model_type = "causal"
    model, tokenizer = load_llama(model_type=model_type)# Load the saved dictionary
    loaded_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/raw/attention_data.pt")

    # Extract each component
    attention = loaded_data['attention']
    input_ids = loaded_data['input_ids']
    word_mappings = loaded_data['word_mappings']
    prompt_len = loaded_data['prompt_len']

    compute_sensitivity(model, tokenizer, attention, input_ids[0])


