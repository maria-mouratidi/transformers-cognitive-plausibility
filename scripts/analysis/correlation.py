import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from scripts.probing.raw import subset
from scripts.visuals.eda import *
from scripts.visuals.normality_test import *
from scripts.visuals.corr_plots import *

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']

def load_processed_data():
    human_df = pd.read_csv('data/task2/processed/processed_participants.csv')
    if subset:
        human_df = human_df[human_df['Sent_ID'] < subset]  # Subset
    model_data = torch.load("/scratch/7982399/thesis/outputs/attention_processed.pt")
    attention = model_data['attention_processed'].cpu()
    return human_df, attention

def map_token_indices(human_df, max_seq_len):
    token_indices = []
    for sent_id in human_df['Sent_ID'].unique():
        sent_tokens = human_df[human_df['Sent_ID'] == sent_id]
        for _, token_row in sent_tokens.iterrows():
            flattened_idx = sent_id * max_seq_len + token_row['Word_ID']
            token_indices.append(flattened_idx)
    return np.array(token_indices)

def correlation_analysis(attention_nonpadded, human_df):
    feature_matrix = human_df[FEATURES].to_numpy()
    results = []
    for layer_idx in range(attention_nonpadded.shape[0]):
        attention_values = attention_nonpadded[layer_idx].numpy()
        for feature_idx, feature_name in enumerate(FEATURES):
            human_feature_values = feature_matrix[:, feature_idx]
            if len(attention_values) > 1:
                # Pearson's correlation
                pearson_r, pearson_p = pearsonr(attention_values, human_feature_values)
                # Spearman's correlation
                spearman_r, spearman_p = spearmanr(attention_values, human_feature_values)
            else:
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

def exploratory_analysis_for_layers(human_df, attention_flat, token_indices, layers_to_analyze, save_dir=None):
    for layer_idx in layers_to_analyze:
        print(f"\n--- Exploratory analysis for Layer {layer_idx} ---")
        layer_attention = attention_flat[layer_idx, token_indices].numpy()
        plot_hist_kde_box(human_df, layer_attention, FEATURES, layer_idx, save_dir)
        plt.close()
        #shapiro_test(human_df, layer_attention, FEATURES, layer_idx)
        plot_regplots(human_df, layer_attention, FEATURES, layer_idx, save_dir)
        plt.close()

def run_full_analysis():
    human_df, attention = load_processed_data()
    num_layers, _, max_seq_len = attention.shape
    token_indices = map_token_indices(human_df, max_seq_len)
    attention_flat = attention.reshape(num_layers, -1)
    attention_nonpadded = attention_flat[:, token_indices]
    
    # --- Exploratory Analysis ---
    first_layer = 0
    middle_layer = num_layers // 2
    last_layer = num_layers - 1
    layers_to_analyze = [first_layer, middle_layer, last_layer]
    save_dir = "outputs/analysis_plots"
    exploratory_analysis_for_layers(human_df, attention_flat, token_indices, layers_to_analyze, save_dir)
    
    # --- Correlation Analysis across ALL layers ---
    results_df = correlation_analysis(attention_nonpadded, human_df)
    
    # --- Global Heatmaps ---
    plot_feature_corr(results_df, 'pearson', save_dir)
    plot_eyegaze_corr(human_df, FEATURES, save_dir)
    
    # --- Filter & print significant correlations ---
    significance_threshold = 0.05
    sig = results_df[
        (results_df['pearson_p_value'] < significance_threshold) &
        (results_df['spearman_p_value'] < significance_threshold)
    ]
    print(sig)

if __name__ == "__main__":
    run_full_analysis()
