import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from scripts.analysis.correlation import load_processed_data
from scripts.visuals.corr_plots import plot_feature_corr
from scripts.visuals.pca_plots import plot_explained_variance, plot_pca_loadings

# TODO: remove noninformtive features? (e.g., go past time)

def apply_pca(human_df, features, n_components=None, variance_threshold=0.95):
    pca = PCA()
    pca_features = pca.fit_transform(human_df[features])
    
    # Determine number of components based on explained variance
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
    for layer_idx in range(attention_nonpadded.shape[0]):
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

def run_analysis_with_pca(task: str, attn_method:str, variance_threshold=0.95, save_dir=None):
    human_df, attention = load_processed_data(attn_method=attn_method, task=task)
    
    token_indices = list(zip(human_df['Sent_ID'], human_df['Word_ID']))
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    
    attention_nonpadded = attention[:, sent_ids, word_ids]
    print("Attention shape: ", attention_nonpadded.shape)
    
    # Apply PCA with dynamic component selection
    pca_df, pca, explained_variance, cumulative_variance = apply_pca(human_df, ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT'])
    
    # Correlation analysis with PCA components
    results_df = correlation_analysis(attention_nonpadded, pca_df)
    
    # Plot PCA-related visuals
    plot_explained_variance(cumulative_variance, save_dir=save_dir)
    plot_pca_loadings(pca, ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT'], save_dir=save_dir)
    plot_feature_corr(results_df, method='pearson', pca=True, save_dir=save_dir)
    plot_feature_corr(results_df, method='spearman', pca=True, save_dir=save_dir)
    
    return results_df

if __name__ == "__main__":
    task = "task2"
    attn_method = "flow"
    save_dir = f"outputs/{task}/{attn_method}/pca"
    run_analysis_with_pca(task=task, attn_method=attn_method, variance_threshold=0.95, save_dir=save_dir)