import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import pickle
from scripts.probing.raw import subset
from scripts.visuals.eda import *
from scripts.visuals.normality_test import *
from scripts.visuals.corr_plots import *

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']

def load_processed_data(attn_method: str, task: str):
    human_df = pd.read_csv(f'data/{task}/processed/processed_participants.csv')
    print(f"\nTotal NaN values: {human_df.isna().sum().sum()}")
    human_df.fillna(0, inplace=True) #TODO: temporary fix for NaN values
    if attn_method == "raw":
        model_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/{attn_method}/attention_data.pt")
        attention = model_data['attention_processed'].cpu()
    elif attn_method == "flow":
        attention = torch.load(f"/scratch/7982399/thesis/outputs/{task}/{attn_method}/attention_flow_processed.pt")
        attention = torch.unsqueeze(attention, 0) 
    elif attn_method == "saliency":
        with open(f"/scratch/7982399/thesis/outputs/{task}/{attn_method}/saliency_data.pkl", 'rb') as f:
            attention = pickle.load(f)
            max_len = max(att.shape[0] for att in attention)
            attention = np.array([np.pad(arr, (0, max_len - arr.shape[0]), mode='constant') for arr in attention]) # pad to max sentence length and stack
            attention = np.nan_to_num(attention, nan=0.0)  # Replace NaN with 0.0
            attention = torch.tensor(attention)
            attention = torch.unsqueeze(attention, 0)  # Add a dummy layer dimension
    return human_df, attention

def map_token_indices(human_df):
    token_indices = [(row['Sent_ID'], row['Word_ID']) for _, row in human_df.iterrows()]
    return token_indices

def correlation_analysis(attention_nonpadded, human_df):
    feature_matrix = human_df[FEATURES].to_numpy()
    results = []
    for layer_idx in range(attention_nonpadded.shape[0]):
        attention_values = attention_nonpadded[layer_idx]
        for feature_idx, feature_name in enumerate(FEATURES):
            human_feature_values = feature_matrix[:, feature_idx]
            if len(attention_values) > 1:
                # Pearson's correlation
                pearson_r, pearson_p = pearsonr(attention_values, human_feature_values)
                # Spearman's correlation
                spearman_r, spearman_p = spearmanr(attention_values, human_feature_values)
            else:
                print("nan values in attention")
                pearson_r, pearson_p = (np.nan, np.nan)
                spearman_r, spearman_p = (np.nan, np.nan)
            results.append({
                'layer': layer_idx,
                'feature': feature_name,
                'pearson_r': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p
            })
    return pd.DataFrame(results)

def exploratory_analysis_for_layers(human_df, attention, layers_to_analyze, save_dir=None):
    for layer_idx in layers_to_analyze:
        print(f"\n--- Exploratory analysis for Layer {layer_idx} ---")
        plot_hist_kde_box(human_df, attention[layer_idx, :], FEATURES, layer_idx, save_dir)
        #shapiro_test(human_df, layer_attention, FEATURES, layer_idx)
        plot_regplots(human_df, attention[layer_idx, :], FEATURES, layer_idx, save_dir)

def run_full_analysis(attn_method: str, task: str,):
    human_df, attention = load_processed_data(attn_method=attn_method, task=task)

    print(f"Attention shape: {attention.shape}")

    token_indices = map_token_indices(human_df)
    
    # Convert token indices to tensors for efficient indexing
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)

    # Efficient batch indexing in PyTorch
    attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
    print(f"Attention non-padded shape: {attention_nonpadded.shape}")
    nan_indices_nonpadded = np.argwhere(np.isnan(attention_nonpadded))
    if nan_indices_nonpadded.size > 0:
        print(f"NaN values detected in attention_nonpadded at indices: {nan_indices_nonpadded}")
        for idx in nan_indices_nonpadded:
            print(f"Specific value at index {idx}: {attention_nonpadded[tuple(idx)]}")
    
    # --- Exploratory Analysis ---
    layers_to_analyze = [0, 15, 31] if attn_method == "raw" else []
    save_dir = f"outputs/{task}/{attn_method}/analysis_plots"
    os.makedirs(save_dir, exist_ok=True)
    #save_dir = None
    print(f"Attention non-padded shape: {attention_nonpadded.shape}")
    exploratory_analysis_for_layers(human_df, attention_nonpadded, layers_to_analyze, save_dir)
    
    # --- Correlation Analysis across ALL layers ---
    results_df = correlation_analysis(attention_nonpadded, human_df)
    
    # --- Global Heatmaps ---
    plot_feature_corr(results_df, method = 'pearson', save_dir = save_dir)
    plot_feature_corr(results_df, method ='spearman', save_dir = save_dir)
    plot_eyegaze_corr(human_df, FEATURES, save_dir)
    
    # --- Filter & print significant correlations ---
    significance_threshold = 0.05
    sig = results_df[
        (results_df['pearson_p_value'] < significance_threshold) &
        (results_df['spearman_p_value'] < significance_threshold)
    ]
    sig.to_csv(f"{save_dir}/significant_correlations.csv", index=False)

if __name__ == "__main__": 
    run_full_analysis(attn_method = "saliency", task="task3")
