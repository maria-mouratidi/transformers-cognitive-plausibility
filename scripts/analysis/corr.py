import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from sklearn.decomposition import PCA
from scripts.analysis.load_attention import load_processed_data
from scripts.visuals.corr_plots import plot_corr, plot_gaze_intercorr
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD']

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
            pearson_r, pearson_p = pearsonr(attention_values, human_feature_values)
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

def apply_pca(human_df, features, n_components=None, variance_threshold=0.95):
    pca = PCA()
    pca_features = pca.fit_transform(human_df[features])
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    if n_components is None:
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Selected {n_components} components to retain {variance_threshold * 100}% variance")
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(human_df[features])
    pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(n_components)])
    explained_variance = pca.explained_variance_ratio_
    return pca_df, pca, explained_variance, cumulative_variance

def pca_correlation_analysis(attention_nonpadded, pca_df):
    results = []
    num_layers = attention_nonpadded.shape[0]
    for layer_idx in range(num_layers):
        attention_values = attention_nonpadded[layer_idx]
        for pc_idx in range(pca_df.shape[1]):
            pc_values = pca_df.iloc[:, pc_idx]
            pearson_r, pearson_p = pearsonr(attention_values, pc_values)
            spearman_r, spearman_p = spearmanr(attention_values, pc_values)
            results.append({
                'layer': layer_idx,
                'principal_component': pca_df.columns[pc_idx],
                'pearson_r': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p
            })
    return pd.DataFrame(results)

def run_full_analysis(model_name: str, attn_methods: list):
    all_results = []
    all_pca_results = []
    for task in tasks:
        for attn_method in attn_methods:
            human_df, attention, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
            
            # Match attention to human_df
            token_indices = map_token_indices(human_df)
            sent_ids, word_ids = zip(*token_indices)
            sent_ids = torch.tensor(sent_ids, dtype=torch.long)
            word_ids = torch.tensor(word_ids, dtype=torch.long)
            attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
            
            # --- Correlation Analysis ---
            results_df = correlation_analysis(attention_nonpadded, human_df)
            pca_df, _, _, _ = apply_pca(human_df, FEATURES, n_components=1) # 1 component is sufficient according to PCA-gaze correlations
            pca_results_df = pca_correlation_analysis(attention_nonpadded, pca_df)
            results_df['attn_method'] = attn_method
            pca_results_df['attn_method'] = attn_method
            results_df['task'] = task
            pca_results_df['task'] = task
            all_results.append(results_df)
            all_pca_results.append(pca_results_df)
    
    # --- Combined Results ---
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_pca_results = pd.concat(all_pca_results, ignore_index=True)
    plot_corr(combined_results, combined_pca_results, attn_method=None, method='spearman', save_dir=f"outputs/combined_corrs_{model_name}.png", significance_threshold=0.05)

    # --- Feature Correlation Analysis ---
    all_FEATURES = FEATURES + ['SFD', 'GPT']  # Full feature list
    pca_df, _, _, _ = apply_pca(human_df, all_FEATURES, variance_threshold=0.95)
    plot_gaze_intercorr(human_df, pca_df, all_FEATURES, save_dir="outputs") # Plot all features for exploration

if __name__ == "__main__":
    llm_models = ["llama", "bert"]
    tasks = ["task2", "task3"]
    attn_methods = ["raw", "flow", "saliency"]

    for model_name in llm_models:
        print(f"--- Combined analysis for {model_name} (excluding raw) ---")
        run_full_analysis(model_name=model_name, attn_methods=attn_methods)