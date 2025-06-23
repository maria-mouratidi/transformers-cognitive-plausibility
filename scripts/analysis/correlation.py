import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from sklearn.decomposition import PCA
from scripts.analysis.load_attention import load_processed_data
from scripts.visuals.corr_plots import plot_other_corr, plot_raw_corr, plot_text_attn_corr

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
            spearman_r, spearman_p = spearmanr(attention_values, human_feature_values)
            row = {
                'feature': 'F' if feature_name == 'nFixations' else 'mPS' if feature_name == 'meanPupilSize' else feature_name,
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
            attention_all_methods = {}
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
                pca_df, _, _, _ = apply_pca(human_df, FEATURES, n_components=1)
                pca_results_df = pca_correlation_analysis(attention_nonpadded, pca_df)
                results_df['attn_method'] = attn_method
                pca_results_df['attn_method'] = attn_method
                results_df['task'] = task
                pca_results_df['task'] = task
                results_df['llm_model'] = model_name
                pca_results_df['llm_model'] = model_name
                all_gaze_results.append(results_df)
                all_pca_results.append(pca_results_df)

                attention_all_methods[attn_method] = attention_nonpadded

            # --- Text Feature Correlation Heatmap ---
            text_feat_path = f"materials/text_features_{task}_{model_name}.csv"
            text_features_df = pd.read_csv(text_feat_path)
            text_features_df['role'] = text_features_df['role'].map({'function': 0, 'content': 1})

            plot_text_attn_corr(
                attention_all_methods,
                text_features_df,
                f"text_corrs_{task}_{model_name}",
                save_dir="outputs",
                model_name=model_name,
            )

    # --- Combined Results ---
    all_gaze_results = pd.concat(all_gaze_results, ignore_index=True)
    all_pca_results = pd.concat(all_pca_results, ignore_index=True)
    combined_df = combine_results(all_gaze_results, all_pca_results)
    plot_raw_corr(combined_df, save_dir="outputs")
    plot_other_corr(combined_df, save_dir="outputs")

    # --- Feature Correlation Analysis ---
    all_FEATURES = FEATURES + ['SFD', 'GPT']
    pca_df, _, _, _ = apply_pca(human_df, all_FEATURES, variance_threshold=0.95)
    #plot_gaze_intercorr(human_df, pca_df, all_FEATURES, save_dir="outputs")

if __name__ == "__main__":
    run_full_analysis()