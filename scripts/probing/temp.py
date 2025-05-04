import torch
import os

data = torch.load(f"/scratch/7982399/thesis/outputs/task2/raw/attention_data.pt")
batch_size = data['input_ids'].shape[0]

i = 2  # Save only the first 2 batches

attention = data['attention'][:, :i].clone().detach()        # Deep copy
input_ids = data['input_ids'][:i].clone().detach()
word_mappings = data['word_mappings'][:i]                    # This is a list, safe to slice
prompt_len = data['prompt_len']                              # Scalar or small value

save_dict = {
    'attention': attention,
    'input_ids': input_ids,
    'word_mappings': word_mappings,
    'prompt_len': prompt_len
}

torch.save(save_dict, "/scratch/7982399/thesis/outputs/task2/raw/batch_data.pt")
