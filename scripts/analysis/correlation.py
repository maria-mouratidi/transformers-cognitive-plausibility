import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from scripts.probing.raw import subset
from visuals.eda import plot_hist_kde_box
from visuals.normality_test import shapiro_test
from visuals.corr_plots import plot_regplots, plot_corr_heatmap

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'WordLen']

def load_processed_data():
    human_df = pd.read_csv('data/task2/processed/averaged_participants.csv')
    human_df = human_df[human_df['Sent_ID'] < subset]  # Subset
    model_data = torch.load("/scratch/7982399/thesis/outputs/attention_processed.pt")
    attention = model_data['attention_processed']
    return human_df, attention

def map_token_indices(human_df, max_seq_len):
    token_indices = []
    for sent_id in human_df['Sent_ID'].unique():
        sent_tokens = human_df[human_df['Sent_ID'] == sent_id]
        for _, token_row in sent_tokens.iterrows():
            flattened_idx = sent_id * max_seq_len + token_row['Word_ID']
            token_indices.append(flattened_idx)
    return np.array(token_indices)

def extract_layer_attention(attention, token_indices, layer_idx):
    attention_flat = attention.reshape(attention.shape[0], -1)
    layer_attention = attention_flat[layer_idx, token_indices].numpy()
    return attention_flat, layer_attention

def exploratory_analysis(human_df, layer_attention_values, layer_idx):
    plot_hist_kde_box(human_df, layer_attention_values, FEATURES, layer_idx)
    shapiro_test(human_df, layer_attention_values, FEATURES, layer_idx)
    plot_regplots(human_df, layer_attention_values, FEATURES, layer_idx)
    plot_corr_heatmap(human_df, layer_attention_values, FEATURES, layer_idx)

def correlation_analysis(attention_nonpadded, human_df):
    feature_matrix = human_df[FEATURES].to_numpy()
    results = []
    for layer_idx in range(attention_nonpadded.shape[0]):
        attention_values = attention_nonpadded[layer_idx].numpy()
        for feature_idx, feature_name in enumerate(FEATURES):
            human_feature_values = feature_matrix[:, feature_idx]
            if len(attention_values) > 1:
                r_value, p_value = pearsonr(attention_values, human_feature_values)
            else:
                r_value, p_value = (np.nan, np.nan)
            results.append({
                'layer': layer_idx,
                'feature': feature_name,
                'pearson_r': r_value,
                'p_value': p_value
            })
    return pd.DataFrame(results)

def main():
    human_df, attention = load_processed_data()
    num_layers, _, max_seq_len = attention.shape
    
    token_indices = map_token_indices(human_df, max_seq_len)
    
    # --- Exploratory analysis for single layer ---
    layer_idx = 0  # customize as needed
    attention_flat, layer_attention_values = extract_layer_attention(attention, token_indices, layer_idx)
    exploratory_analysis(human_df, layer_attention_values, layer_idx)
    
    # --- Correlation Analysis ---
    attention_nonpadded = attention_flat[:, token_indices]
    results_df = correlation_analysis(attention_nonpadded, human_df)
    
    print(results_df)

if __name__ == "__main__":
    main()
