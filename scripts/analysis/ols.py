import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
from statsmodels.api import OLS, add_constant
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from scripts.analysis.correlation import load_processed_data, map_token_indices
from scripts.analysis.correlation_pca import apply_pca

# ------------------ Load and Prepare Data ------------------

attn_method, task, model_name = "saliency", "task3", "llama"
text_df = pd.read_csv(f'materials/text_features_{task}.csv')
text_df['role'] = text_df['role'].map({'function': 0, 'content': 1})
gaze_df, attention_tensor, save_dir  = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)

X_text = text_df[['frequency', 'length', 'surprisal', 'role']]
gaze_features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']
y_gaze = gaze_df[gaze_features]

y_pca, pca_obj, explained_var, cum_var = apply_pca(gaze_df, gaze_features)

# ------------------ Attention Features ------------------
sent_idx, word_idx = zip(*map_token_indices(gaze_df))
sent_idx = np.array(sent_idx)
word_idx = np.array(word_idx)

selected_layers = [1, 31] if (model_name == "llama" and attn_method == "raw") else [0]
attention_features = []
for layer in selected_layers:
    layer_attention = attention_tensor[layer, sent_idx, word_idx]
    attention_features.append(layer_attention)

X_attention = np.column_stack(attention_features)
attention_feature_names = [f'attention_layer_{i}' for i in selected_layers]

X_text_attn = X_text.copy()
for idx, name in enumerate(attention_feature_names):
    X_text_attn[name] = X_attention[:, idx]

# ------------------ Train-Test Split ------------------
(X_train_text, X_test_text, 
 y_train_gaze, y_test_gaze, 
 y_train_pca, y_test_pca, 
 X_train_attn, X_test_attn) = train_test_split(
    X_text, y_gaze, y_pca, X_text_attn, test_size=0.2, random_state=42
)

# ------------------ OLS Models ------------------
model_defs = [
    ("TextOnly_Gaze", X_train_text, X_test_text, y_train_gaze, y_test_gaze, list(X_text.columns)),
    ("TextOnly_PCA", X_train_text, X_test_text, y_train_pca, y_test_pca, list(X_text.columns)),
    ("TextAttn_PCA", X_train_attn, X_test_attn, y_train_pca, y_test_pca, list(X_text_attn.columns)),
    ("TextAttn_Gaze", X_train_attn, X_test_attn, y_train_gaze, y_test_gaze, list(X_text_attn.columns)),
]

model_performance = []
feature_importance = []
r2_dict = {}

for model_name, X_train, X_test, y_train, y_test, feat_names in model_defs:
    # If y is DataFrame (multioutput), fit one OLS per column
    if isinstance(y_train, pd.DataFrame) or (hasattr(y_train, 'shape') and len(y_train.shape) > 1 and y_train.shape[1] > 1):
        for idx, y_col in enumerate(y_train.columns):
            y_train_col = y_train[y_col]
            y_test_col = y_test[y_col]
            X_train_const = add_constant(X_train)
            X_test_const = add_constant(X_test)
            ols_model = OLS(y_train_col, X_train_const).fit()
            y_pred = ols_model.predict(X_test_const)
            r2 = ols_model.rsquared
            n = len(y_test_col)
            p = X_test_const.shape[1] - 1
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            rmse = np.sqrt(np.mean((y_test_col - y_pred) ** 2))
            model_id = f"{model_name}_{y_col}"
            r2_dict[model_id] = r2
            model_performance.append({
                "task": task,
                "LLM_model": model_name,
                "attention_method": attn_method,
                "OLS_model": model_id,
                "rsquared": r2,
                "rsquared_adj": r2_adj,
                "rmse": rmse
            })
            for feat in X_train_const.columns:
                coef = ols_model.params[feat]
                tval = ols_model.tvalues[feat]
                stderr = ols_model.bse[feat]
                pval = ols_model.pvalues[feat]
                feat_vals = X_train_const[feat] if feat != "const" else np.ones(len(X_train_const))
                q25, q50, q75 = np.percentile(feat_vals, [25, 50, 75])
                feature_importance.append({
                    "task": task,
                    "LLM_model": model_name,
                    "attention_method": attn_method,
                    "OLS_model": model_id,
                    "feature_name": feat,
                    "coefficient": coef,
                    "t": tval,
                    "std_error": stderr,
                    "p_value": pval,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75
                })
    else:
        # Single output
        X_train_const = add_constant(X_train)
        X_test_const = add_constant(X_test)
        ols_model = OLS(y_train, X_train_const).fit()
        y_pred = ols_model.predict(X_test_const)
        r2 = ols_model.rsquared
        n = len(y_test)
        p = X_test_const.shape[1] - 1
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        model_id = model_name
        r2_dict[model_id] = r2
        model_performance.append({
            "task": task,
            "LLM_model": model_name,
            "attention_method": attn_method,
            "OLS_model": model_id,
            "rsquared": r2,
            "rsquared_adj": r2_adj,
            "rmse": rmse
        })
        for feat in X_train_const.columns:
            coef = ols_model.params[feat]
            tval = ols_model.tvalues[feat]
            stderr = ols_model.bse[feat]
            pval = ols_model.pvalues[feat]
            feat_vals = X_train_const[feat] if feat != "const" else np.ones(len(X_train_const))
            q25, q50, q75 = np.percentile(feat_vals, [25, 50, 75])
            feature_importance.append({
                "task": task,
                "LLM_model": model_name,
                "attention_method": attn_method,
                "model_name": model_id,
                "feature_name": feat,
                "coefficient": coef,
                "t": tval,
                "std_error": stderr,
                "p_value": pval,
                "q25": q25,
                "q50": q50,
                "q75": q75
            })

# ------------------ Pairwise Wilcoxon t-test on R² ------------------
model_ids = list(r2_dict.keys())
for i, m1 in enumerate(model_ids):
    for j, m2 in enumerate(model_ids):
        if i >= j:
            continue
        # Wilcoxon expects paired samples, but here we have one R² per model, so this is illustrative
        try:
            stat, p = wilcoxon([r2_dict[m1]], [r2_dict[m2]])
        except Exception:
            stat, p = np.nan, np.nan
        for m_from, m_to in [(m1, m2), (m2, m1)]:
            idx = next((k for k, d in enumerate(model_performance)
                        if d["OLS_model"] == m_from), None)
            if idx is not None:
                key = f"wilcoxon_vs_{m_to}"
                model_performance[idx][key] = p

# ------------------ Save Results ------------------
results_dir = os.path.join(save_dir, "ols")
os.makedirs(results_dir, exist_ok=True)
perf_df = pd.DataFrame(model_performance)
feat_df = pd.DataFrame(feature_importance)
perf_df.to_csv(os.path.join(results_dir, "model_performance.csv"), index=False)
feat_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
print(f"OLS results saved to: {results_dir}")