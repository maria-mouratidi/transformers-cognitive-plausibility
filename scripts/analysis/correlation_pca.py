import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from scripts.analysis.correlation import map_token_indices
from scripts.analysis.correlation import FEATURES
from scripts.analysis.correlation import load_processed_data
from scripts.visuals.corr_plots import plot_feature_corr
from scripts.visuals.pca_plots import plot_explained_variance, plot_pca_loadings
import os

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

def correlation_analysis(attention_nonpadded, pca_df):
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

def run_analysis_with_pca(attn_method: str, task: str, model_name: str, variance_threshold=0.95):
    human_df, attention, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
    save_dir = os.path.join(save_dir, "pca")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Attention shape: {attention.shape}")

    token_indices = map_token_indices(human_df)
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
    print(f"Attention non-padded shape: {attention_nonpadded.shape}")

    # Apply PCA with dynamic component selection
    pca_df, pca, explained_variance, cumulative_variance = apply_pca(human_df, FEATURES, variance_threshold=variance_threshold)

    # Correlation analysis with PCA components
    results_df = correlation_analysis(attention_nonpadded, pca_df)

    # Plot PCA-related visuals
    plot_explained_variance(cumulative_variance, save_dir=save_dir)
    plot_pca_loadings(pca, FEATURES, save_dir=save_dir)
    #plot_feature_corr(results_df, attn_method, method='pearson', pca=True, save_dir=save_dir)
    plot_feature_corr(results_df, attn_method, method='spearman', pca=True, save_dir=save_dir)

    # Save significant correlations
    significance_threshold = 0.05
    sig = results_df[
        (results_df['spearman_p_value'] < significance_threshold)
    ]
    sig.to_csv(f"{save_dir}/significant_pca_correlations.csv", index=False)

    return results_df

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
                run_analysis_with_pca(attn_method=attn_method, task=task, model_name=model_name, variance_threshold=0.95)