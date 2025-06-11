import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from scripts.analysis.load_attention import load_processed_data
from scripts.visuals.hist import *
from scripts.visuals.normality_test import *
from scripts.visuals.corr_plots import *

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']

def map_token_indices(human_df):
    token_indices = [(row['Sent_ID'], row['Word_ID']) for _, row in human_df.iterrows()]
    return token_indices

def correlation_analysis(attention_nonpadded, human_df):
    feature_matrix = human_df[FEATURES].to_numpy()
    results = []
    num_layers = attention_nonpadded.shape[0]
    
    for layer_idx in range(num_layers):
        attention_values = attention_nonpadded[layer_idx]
        
        for feature_idx, feature_name in enumerate(FEATURES):
            human_feature_values = feature_matrix[:, feature_idx]
            # Pearson's correlation
            pearson_r, pearson_p = pearsonr(attention_values, human_feature_values)
            # Spearman's correlation
            spearman_r, spearman_p = spearmanr(attention_values, human_feature_values)

            row = {

                'feature': feature_name,
                'pearson_r': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p,
                'layer': layer_idx
            }

            results.append(row)
    
    return pd.DataFrame(results)


def run_full_analysis(attn_method: str, task: str, model_name: bool):
    
    human_df, attention, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Attention shape: {attention.shape}")

    token_indices = map_token_indices(human_df)
    
    # Convert token indices to tensors for efficient indexing
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)

    # Efficient batch indexing in PyTorch
    attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
    print(f"Attention non-padded shape: {attention_nonpadded.shape}")
    
    # --- Exploratory Analysis ---
    if model_name == "llama" and attn_method == "raw":
        layers_to_analyze = [0, 1, 31]
    elif model_name == "bert" and attn_method == "raw":
        layers_to_analyze = [0, 4, 11]
    else:
        layers_to_analyze = [] 

    for layer_idx in layers_to_analyze:
        plot_regplots(human_df, attention_nonpadded[layer_idx, :], FEATURES, layer_idx, attn_method, save_dir)

    # --- Correlation Analysis across ALL layers ---
    results_df = correlation_analysis(attention_nonpadded, human_df)
    
    # --- Global Heatmaps ---
    #plot_feature_corr(results_df, method = 'pearson', save_dir = save_dir)
    plot_feature_corr(results_df, attn_method, method ='spearman', save_dir = save_dir)
    #plot_eyegaze_corr(human_df, FEATURES, save_dir)
    
    # --- Filter & print significant correlations ---
    significance_threshold = 0.05
    sig = results_df[
        (results_df['pearson_p_value'] < significance_threshold) &
        (results_df['spearman_p_value'] < significance_threshold)
    ]
    sig.to_csv(f"{save_dir}/significant_correlations.csv", index=False)

if __name__ == "__main__": 
    llm_models = ["llama", "bert"]
    tasks = ["task2", "task3"]
    attn_methods = ["raw", "flow", "saliency"]

    for model_name in llm_models:
        for task in tasks:
            for attn_method in attn_methods:
                if model_name == "bert" and attn_method == "flow":
                    print(f"Skipping {model_name}, {task}, {attn_method} due to missing data.")
                    continue
                run_full_analysis(attn_method = attn_method, task=task, model_name=model_name)
