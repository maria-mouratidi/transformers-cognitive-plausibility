import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from sklearn.decomposition import PCA
from scripts.analysis.load_attention import load_processed_data
from scripts.visuals.corr_plots import plot_regplots, plot_combined_corr, plot_feature_intercorr
from scripts.visuals.pca_plots import plot_explained_variance, plot_pca_loadings
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD']

def map_token_indices(human_df):
    token_indices = [(row['Sent_ID'], row['Word_ID']) for _, row in human_df.iterrows()]
    return token_indices

def feature_correlation_analysis(attention_nonpadded, human_df):
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

def plot_feature_pca_intercorr(human_df, pca_df, features, save_dir=None):
    """
    Plot heatmap of inter-correlations among gaze features and PCA components.
    """
    combined = pd.concat([human_df[features], pca_df], axis=1)
    corr_matrix = combined.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0)
    plt.title("Inter-correlations: Eye-gaze Features & PCA Components")
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'feature_pca_intercorr_heatmap.png')
        plt.savefig(save_path)
        print(f"Saved inter-correlation heatmap to {save_path}")

def run_full_analysis(attn_method: str, task: str, model_name: str, variance_threshold=0.95):
    human_df, attention, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Attention shape: {attention.shape}")

    token_indices = map_token_indices(human_df)
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
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

    # --- Feature Correlation Analysis ---
    results_df = feature_correlation_analysis(attention_nonpadded, human_df)
    results_df.to_csv(f"{save_dir}/feature_correlations.csv", index=False)

    # --- PCA Correlation Analysis ---
    pca_df, pca, explained_variance, cumulative_variance = apply_pca(human_df, FEATURES, n_components=1)
    pca_results_df = pca_correlation_analysis(attention_nonpadded, pca_df)
    pca_results_df.to_csv(f"{save_dir}/pca_correlations.csv", index=False)

    # --- Combined Plot ---
    plot_combined_corr(results_df, pca_results_df, attn_method, method='spearman', save_dir=save_dir, significance_threshold=0.05)

    # --- Save significant correlations ---
    significance_threshold = 0.05
    sig = results_df[
        (results_df['pearson_p_value'] < significance_threshold) &
        (results_df['spearman_p_value'] < significance_threshold)
    ]
    sig.to_csv(f"{save_dir}/significant_feature_correlations.csv", index=False)
    sig_pca = pca_results_df[
        (pca_results_df['spearman_p_value'] < significance_threshold)
    ]
    sig_pca.to_csv(f"{save_dir}/significant_pca_correlations.csv", index=False)

    # --- Optional: Plot PCA explained variance and loadings ---
    plot_explained_variance(cumulative_variance, save_dir=save_dir)
    plot_pca_loadings(pca, FEATURES, save_dir=save_dir)
    plot_feature_intercorr(human_df, pca_df, FEATURES, save_dir=save_dir)

if __name__ == "__main__":
    llm_models = ["llama", "bert"]
    tasks = ["task2", "task3"]
    attn_methods = ["raw", "flow", "saliency"]

    for model_name in llm_models:
        for task in tasks:
            for attn_method in attn_methods:
                run_full_analysis(attn_method=attn_method, task=task, model_name=model_name, variance_threshold=0.95)