import torch
import os

# Create output directory if it doesn't exist
output_dir = "/scratch/7982399/thesis/outputs/task2/raw/split_batches"
os.makedirs(output_dir, exist_ok=True)

# Load the full data
data = torch.load(f"/scratch/7982399/thesis/outputs/task2/raw/attention_data.pt")
batch_size = data['input_ids'].shape[0]

# Process batches from 5 to 300 in groups of 10
start_batch = 9
end_batch = 299

for start_idx in range(start_batch, end_batch + 1, 10):
    end_idx = min(start_idx + 9, end_batch)  # Ensure we don't go beyond batch 300
    
    # Create deep copies of the data for this group
    attention = data['attention'][:, start_idx:end_idx+1].clone().detach()
    input_ids = data['input_ids'][start_idx:end_idx+1].clone().detach()
    
    # Create dictionary to save
    save_dict = {
        'attention': attention,
        'input_ids': input_ids,
    }
    
    # Save with appropriate naming
    output_file = f"{output_dir}/batch_{start_idx}-{end_idx}.pt"
    torch.save(save_dict, output_file)
    print(f"Saved batches {start_idx}-{end_idx} to {output_file}")