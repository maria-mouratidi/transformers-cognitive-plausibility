import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from sklearn.decomposition import PCA
from scripts.analysis.load_attention import load_processed_data
from scripts.visuals.corr_plots import plot_other_corr, plot_raw_corr, plot_pca_loadings
from scripts.constants import FEATURES, ALL_FEATURES, CUSTOM_PALETTE, MODEL_TITLES, TASK_TITLES, corr_plt_params, ols_plt_params

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
            spearman_r, spearman_p = spearmanr(attention_values, human_feature_values)
            row = {
                'feature': 'F' if feature_name == 'nFixations' else 'mPS' if feature_name == 'meanPupilSize' else feature_name,
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p,
                'layer': layer_idx
            }
            results.append(row)
    return pd.DataFrame(results)

def apply_pca(human_df, features, task, variance_threshold=0.95):
    # STEP 1: PCA to determine number of components for 95% variance
    pca_full = PCA()
    pca_full.fit(human_df[ALL_FEATURES])
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_required = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(task)
    print(f"{n_components_required} components required to retain {variance_threshold * 100:.1f}% variance.")

    # STEP 2: Plot loadings using full PCA
    plot_pca_loadings(pca_full, features=ALL_FEATURES, filename=f"loadings_exploration_{task}")
    pca_full = PCA(n_components=2)
    pca_full.fit_transform(human_df[ALL_FEATURES])
    print(f"Explained variance for full PCA with 2 component: {pca_full.explained_variance_ratio_[0]:.4f}")
   
    # STEP 3: Run PCA with only 1 component
    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(human_df[features])
    pca_df = pd.DataFrame(pca_features, columns=['PC1'])
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance for 1 component: {explained_variance[0]:.4f}\n\n")

    return pca_df, pca, explained_variance, cumulative_variance

def pca_correlation_analysis(attention_nonpadded, pca_df):
    results = []
    num_layers = attention_nonpadded.shape[0]
    for layer_idx in range(num_layers):
        attention_values = attention_nonpadded[layer_idx]
        for pc_idx in range(pca_df.shape[1]):
            pc_values = pca_df.iloc[:, pc_idx]
            spearman_r, spearman_p = spearmanr(attention_values, pc_values)
            results.append({
                'layer': layer_idx,
                'principal_component': pca_df.columns[pc_idx],
                'spearman_r': spearman_r,
                'spearman_p_value': spearman_p
            })
    return pd.DataFrame(results)

def combine_results(results_df, pca_results_df):
    """
    Combine results DataFrames for features and PCA components.
    """
    results_df = results_df.copy()
    pca_results_df = pca_results_df.copy()
    
    # Add type column to distinguish between features and PCA components
    results_df['type'] = 'Feature'
    pca_results_df['type'] = 'PC'
    
    pca_results_df = pca_results_df.rename(columns={'principal_component': 'feature'})
    combined = pd.concat([results_df, pca_results_df], ignore_index=True)
    
    return combined

def run_full_analysis():
    all_gaze_results = []
    all_pca_results = []
    for model_name in ['llama', 'bert']:
        for task in ["task2", "task3"]:
            for attn_method in ["raw", "flow", "saliency"]:
                human_df, attention, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
                
                # Match attention to human_df
                token_indices = map_token_indices(human_df)
                sent_ids, word_ids = zip(*token_indices)
                sent_ids = torch.tensor(sent_ids, dtype=torch.long)
                word_ids = torch.tensor(word_ids, dtype=torch.long)
                attention_nonpadded = attention[:, sent_ids, word_ids].numpy()
                
                # --- Correlation Analysis ---
                results_df = correlation_analysis(attention_nonpadded, human_df)
                pca_df, _, _, _ = apply_pca(human_df, FEATURES, task) # 1 component is sufficient according to PCA-gaze correlations
                pca_results_df = pca_correlation_analysis(attention_nonpadded, pca_df)
                results_df['attn_method'] = attn_method
                pca_results_df['attn_method'] = attn_method
                results_df['task'] = task
                pca_results_df['task'] = task
                results_df['llm_model'] = model_name
                pca_results_df['llm_model'] = model_name
                all_gaze_results.append(results_df)
                all_pca_results.append(pca_results_df)
    
    # --- Combined Results ---
    all_gaze_results = pd.concat(all_gaze_results, ignore_index=True)
    all_pca_results = pd.concat(all_pca_results, ignore_index=True)
    combined_df = combine_results(all_gaze_results, all_pca_results)
    plot_raw_corr(combined_df, save_dir="outputs")
    plot_other_corr(combined_df, save_dir="outputs")

if __name__ == "__main__":
    run_full_analysis()