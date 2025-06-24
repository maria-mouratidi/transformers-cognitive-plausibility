import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split
from scripts.analysis.correlation import load_processed_data, map_token_indices, apply_pca
from scripts.constants import FEATURES
from scripts.analysis.perm_feat_imp import compute_permutation_importance
from scripts.visuals.corr_plots import plot_text_attn_corr
from scripts.visuals.regression_plots import plot_metric, plot_attention_feature_importances

# ------------------ Configurations ------------------
LLM_MODELS = ["llama", "bert"]
TASKS = ["task2", "task3"]
ATTENTION_METHODS = ["raw", "flow", "saliency"]
PREDICTOR_TYPES = ["text", "raw", "flow", "saliency"]
DEPENDENT_TYPES = ["gaze", "pca"]

results = []
predictions = {}  
per_sample_errors = {} 
pca_inv_per_sample_errors = {}

def get_attention_features(attention_tensor, gaze_df, model_name, attention_method):
    """Extract attention features for the given model and attention method."""
    sentence_indices, word_indices = zip(*map_token_indices(gaze_df))
    sentence_indices = np.array(sentence_indices)
    word_indices = np.array(word_indices)
    # For LLaMA raw, use layer 1; otherwise, use layer 0
    selected_layers = [1] if attention_method == "raw" and model_name == "llama" else [0]
    features = [attention_tensor[layer, sentence_indices, word_indices] for layer in selected_layers]
    attention_features = np.column_stack(features)
    attention_feature_names = [f"attention_layer_{i}" for i in selected_layers]
    return attention_features, attention_feature_names

def run_ols(X_train, X_test, y_train, y_test, meta):
    """Fit OLS, store predictions and results, and compute permutation importances."""
    X_train_const = add_constant(X_train)
    X_test_const = add_constant(X_test)

    ols_model = OLS(y_train, X_train_const).fit()
    y_pred = ols_model.predict(X_test_const)
    
    key = (meta["task"], meta["llm_model"], meta["attn_method"], meta["predictors"], meta["dependent"])
    predictions[key] = np.array(y_pred)
    per_sample_errors[key] = (np.array(y_test) - np.array(y_pred)) ** 2
    
    # If this is a PCA model, immediately inverse transform and store per-feature errors
    if meta["dependent"] == "pca" and "pca_model" in meta:
        pca_model = meta["pca_model"]
        y_pred_pca = np.array(y_pred).reshape(-1, 1)
        y_pred_inv = pca_model.inverse_transform(y_pred_pca)
        y_test_gaze = meta.get("y_test_gaze", None)
        for i, dependent in enumerate(FEATURES): # Calculate per-feature errors
            inv_key = (meta["task"], meta["llm_model"], meta["attn_method"], meta["predictors"], dependent)
            y_true = np.array(y_test_gaze[dependent])
            pca_inv_per_sample_errors[inv_key] = (y_true - y_pred_inv[:, i]) ** 2

    r2 = ols_model.rsquared
    n = len(y_test)
    p = X_test_const.shape[1] - 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Permutation importance
    X_test_np = X_test_const.values
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test
    perm_importances = compute_permutation_importance(
        ols_model, X_test_np, y_test_np, X_test_const.columns.tolist(), n_repeats=10)
    perm_imp_dict = {name: (mean_imp, std_imp) for name, mean_imp, std_imp in perm_importances}
    
    meta_keys = ["task", "llm_model", "attn_method", "predictors", "dependent"]
    meta_essentials = {k: meta[k] for k in meta_keys if k in meta}
    
    for feature in X_train_const.columns:
        results.append({
            **meta_essentials,
            "predictors": "text_only" if meta["attn_method"] == "none" else f"text+{meta['attn_method']}",
            "dependent": meta["dependent"],
            "feature_name": feature,
            "coefficient": ols_model.params[feature],
            "t": ols_model.tvalues[feature],
            "std_error": ols_model.bse[feature],
            "p_value": ols_model.pvalues[feature],
            "rsquared": r2, "rsquared_adj": r2_adj,
            "rmse": rmse,
            "f_pvalue": ols_model.f_pvalue,
            "perm_imp_mean": perm_imp_dict.get(feature, (np.nan, np.nan))[0], "perm_imp_std": perm_imp_dict.get(feature, (np.nan, np.nan))[1],
            "wilcoxon_baseline_stat": np.nan, "wilcoxon_baseline_p": np.nan,  
            "wilcoxon_pca_vs_gaze_stat": np.nan, "wilcoxon_pca_vs_gaze_p": np.nan
        })

text_feature_names = ['frequency', 'length', 'role', 'surprisal']
corr_results = []

for task in TASKS:
    for llm_model in LLM_MODELS:
        # Load text features and map role to numeric
        text_features_df = pd.read_csv(f'materials/text_features_{task}_{llm_model}.csv')
        text_features_df['role'] = text_features_df['role'].map({'function': 0, 'content': 1})
        X_text = text_features_df[['frequency', 'length', 'surprisal', 'role']]
        gaze_df, _, _ = load_processed_data(attn_method="raw", task=task, model_name=llm_model)
        y_gaze = gaze_df[FEATURES]
        y_pca, pca_model, _, _ = apply_pca(gaze_df, FEATURES, n_components=1)
        # text-only models
        X_train_text, X_test_text, y_train_gaze, y_test_gaze, y_train_pca, y_test_pca = train_test_split(
            X_text, y_gaze, y_pca, test_size=0.2, random_state=42)
      
        # TextOnly Gaze (baseline) for each model
        for gaze_feature in y_gaze.columns:
            run_ols(
                X_train_text, X_test_text, y_train_gaze[gaze_feature], y_test_gaze[gaze_feature],
                meta={"task": task, "llm_model": llm_model, "attn_method": "none", "predictors": "text_only", "dependent": gaze_feature})
        # TextOnly PCA (baseline) for each model
        run_ols(
            X_train_text, X_test_text, y_train_pca, y_test_pca.squeeze(),
            meta={"task": task, "llm_model": llm_model, "attn_method": "none", "predictors": "text_only", "dependent": "pca", "pca_model": pca_model, "y_test_gaze": y_test_gaze})

        for attention_method in ATTENTION_METHODS:
            # Load attention and gaze for this method
            _, attention_tensor, _ = load_processed_data(attn_method=attention_method, task=task, model_name=llm_model)
            attention_features, attention_feature_names = get_attention_features(attention_tensor, gaze_df, llm_model, attention_method)
            X_text_attn = X_text.copy()
            for idx, name in enumerate(attention_feature_names):
                X_text_attn[name] = attention_features[:, idx]

            X_train_attn, X_test_attn, y_train_gaze, y_test_gaze, y_train_pca, y_test_pca = train_test_split(
                X_text_attn, y_gaze, y_pca, test_size=0.2, random_state=42)
            
            for name in attention_feature_names:
                X_train_attn[name] = X_train_attn[name].values
                X_test_attn[name] = X_test_attn[name].values
            
            # Attention Gaze models
            for gaze_feature in y_gaze.columns:
                run_ols(
                    X_train_attn, X_test_attn, y_train_gaze[gaze_feature], y_test_gaze[gaze_feature],
                    meta={"task": task, "llm_model": llm_model, "attn_method": attention_method, "predictors": f"text+{attention_method}", "dependent": gaze_feature})
            # Attention PCA model
            run_ols(
                X_train_attn, X_test_attn, y_train_pca, y_test_pca.squeeze(),
                meta={"task": task, "llm_model": llm_model, "attn_method": attention_method, "predictors": f"text+{attention_method}", "dependent": "pca", "pca_model": pca_model, "y_test_gaze": y_test_gaze}
            )
            # Correlation between text features and attention
            for text_feature in text_feature_names:
                text_feature_values = text_features_df[text_feature].values
                r, pvalue = scipy.stats.spearmanr(attention_features, text_feature_values)
                corr_results.append({
                    "task": task,
                    "llm_model": llm_model,
                    "attn_method": attention_method,
                    "text_feature": text_feature,
                    "spearman_r": r,
                    "spearman_p_value": pvalue
                })

# Save unified results
results_df = pd.DataFrame(results)

# --- Wilcoxon tests ---
for idx, row in results_df[results_df['feature_name']=='const'].iterrows():  #index only for one feature, since rmses are the same for all features in a model
    key = (row["task"], row["llm_model"], row["attn_method"], row["predictors"], row["dependent"])
    # 1. Wilcoxon Attn vs text-only baseline (same llm, task, dependent) (per sample RMSE comparison)
    if row["predictors"] != "text_only":
        base_key = (row["task"], row["llm_model"], "none", "text_only", row["dependent"])
        if key in per_sample_errors and base_key in per_sample_errors:
            stat_rmse, p_rmse = scipy.stats.wilcoxon(per_sample_errors[key], per_sample_errors[base_key])
            results_df.at[idx, "wilcoxon_baseline_stat"] = stat_rmse
            results_df.at[idx, "wilcoxon_baseline_p"] = p_rmse
    # 2. Wilcoxon PCA model vs Gaze model (inverted PCA predictions vs. Gaze predictions, per feature)
    # (same llm, task, attn, predictors)s
    if row["dependent"] != "pca":
        stat, p = scipy.stats.wilcoxon(per_sample_errors[key], pca_inv_per_sample_errors[key]) #same key since we match 1:1 errors of inv pca to a specific gaze feature
        results_df.at[idx, "wilcoxon_pca_vs_gaze_stat"] = stat
        results_df.at[idx, "wilcoxon_pca_vs_gaze_p"] = p


corr_df = pd.DataFrame(corr_results)
results_df.to_csv("outputs/ols_unified_performance.csv", index=False)
# corr_df.to_csv("outputs/text_attn_corr.csv", index=False)

# Plotting
plot_metric(results_df, "rsquared_adj", "outputs/ols_r2.pdf")
plot_attention_feature_importances(results_df, "outputs/ols_feature_importances")
plot_text_attn_corr(corr_df, save_path="outputs/text_attn_corr.pdf")