import pandas as pd
import torch
import pickle
import numpy as np
import os

def load_processed_data(attn_method: str, task: str, model_name: str = "llama"):
    if task == "task2":
        prompt_len = 24
    elif task == "task3":
        prompt_len = 11
    
    human_df = pd.read_csv(f'data/{task}/processed/processed_participants.csv')
    print(f"Total NaN values: {human_df.isna().sum().sum()}")
    human_df.fillna(0, inplace=True) # temporary fix for NaN values in the dataframe

    if attn_method == "raw":
        model_data = torch.load(f"/scratch/7982399/thesis/outputs/{attn_method}/{task}/{model_name}/attention_processed.pt", map_location="cpu")  
        attention = model_data['attention_processed'].cpu()
    
    elif attn_method == "flow":
        attention = torch.load(f"/scratch/7982399/thesis/outputs/{attn_method}/{task}/{model_name}/attention_flow.pt")
        attention = torch.unsqueeze(attention, 0).cpu()
    
    elif attn_method == "saliency":
        with open(f"/scratch/7982399/thesis/outputs/{attn_method}/{task}/{model_name}/saliency_data.pkl", 'rb') as f:
            attention = pickle.load(f)
            max_len = max(att.shape[0] for att in attention)
            attention = np.array([np.pad(arr, (0, max_len - arr.shape[0]), mode='constant') for arr in attention]) # pad to max sentence length and stack
            attention = np.nan_to_num(attention, nan=0.0)  # Replace NaN with 0.0
            attention = torch.tensor(attention)
            attention = torch.unsqueeze(attention, 0)  # Add a dummy layer dimension
            attention = attention[:, :, prompt_len:] 
    def get_save_dir(base="outputs", subdir=""):
        path_parts = [base, attn_method, task, model_name, subdir]
        return os.path.join(*path_parts)
    
    return human_df, attention, get_save_dir()
