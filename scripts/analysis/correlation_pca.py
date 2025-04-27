import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# TODO: remove noninformtive features? (e.g., go past time)

def apply_pca(human_df, features, n_components=2, variance_threshold=0.95):
    pca = PCA()
    pca_features = pca.fit_transform(human_df[features])
    
    # Determine number of components based on explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    #n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    #print(f"Selected {n_components} components to retain {variance_threshold * 100}% variance")
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

def plot_explained_variance(cumulative_variance, save_dir=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs Number of Components")
    plt.legend()
    plt.grid()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/explained_variance.png")
    plt.show()

def plot_pca_loadings(pca, features, save_dir=None):
    loadings = pd.DataFrame(pca.components_, columns=features, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Component Loadings")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/pca_loadings.png")
    plt.show()

def plot_pca_correlations(results_df, method='pearson', save_dir=None):
    pivot = results_df.pivot(index='layer', columns='principal_component', values=f'{method}_r')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"Layer-wise Attention vs PCA Components Correlations ({method.title()})")
    plt.xlabel("Principal Components")
    plt.ylabel("Transformer Layer")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/pca_corr_{method}.png")
    plt.show()

def run_analysis_with_pca(task: str, variance_threshold=0.95, save_dir=None):
    human_df = pd.read_csv('data/task2/processed/processed_participants.csv')
    model_data = torch.load(f"/scratch/7982399/thesis/outputs/{task}/{attn_method}/attention_processed.npy")
    attention = model_data['attention_processed'].cpu()
    
    token_indices = list(zip(human_df['Sent_ID'], human_df['Word_ID']))
    sent_ids, word_ids = zip(*token_indices)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    
    attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
    
    # Apply PCA with dynamic component selection
    pca_df, pca, explained_variance, cumulative_variance = apply_pca(human_df, ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT'], variance_threshold)
    
    # Correlation analysis with PCA components
    results_df = correlation_analysis(attention_nonpadded, pca_df)
    
    # Plot PCA-related visuals
    plot_explained_variance(cumulative_variance, save_dir=save_dir)
    plot_pca_loadings(pca, ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT'], save_dir=save_dir)
    plot_pca_correlations(results_df, method='pearson', save_dir=save_dir)
    plot_pca_correlations(results_df, method='spearman', save_dir=save_dir)
    
    return results_df

if __name__ == "__main__":
    task = "task2"
    attn_method = "flow"
    save_dir = f"outputs/{task}/{attn_method}/pca"
    run_analysis_with_pca(task=task, attn_method=attn_method, variance_threshold=0.95, save_dir=save_dir)