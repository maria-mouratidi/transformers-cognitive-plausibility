import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.analysis.correlation import load_processed_data, map_token_indices, apply_pca, FEATURES
from scripts.analysis.perm_feat_imp import compute_permutation_importance
# ------------------ Configurations ------------------
llm_models = ["llama", "bert"]
tasks = ["task2", "task3"]
attn_methods = ["raw", "flow", "saliency"]
predictor_types = ["text", "raw", "flow", "saliency"]
dependent_types = ["gaze", "pca"]

results = []

def get_attention_features(attention_tensor, gaze_df, model_name, attn_method):
    sent_idx, word_idx = zip(*map_token_indices(gaze_df))
    sent_idx = np.array(sent_idx)
    word_idx = np.array(word_idx)
    if attn_method == "raw" and model_name == "llama":
        selected_layers = [1, 31]
    else:
        selected_layers = [0]
    features = []
    for layer in selected_layers:
        features.append(attention_tensor[layer, sent_idx, word_idx])
    X_attention = np.column_stack(features)
    attn_names = [f"attention_layer_{i}" for i in selected_layers]
    return X_attention, attn_names

def scale_text_features(X_train, X_test):
    scaler = StandardScaler()
    cols = ['frequency', 'length', 'surprisal']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cols] = scaler.fit_transform(X_train[cols])
    X_test_scaled[cols] = scaler.transform(X_test[cols])
    return X_train_scaled, X_test_scaled

def run_ols(X_train, X_test, y_train, y_test, predictors, meta):
    X_train_const = add_constant(X_train)
    X_test_const = add_constant(X_test)
    ols_model = OLS(y_train, X_train_const).fit()
    y_pred = ols_model.predict(X_test_const)
    r2 = ols_model.rsquared
    n = len(y_test)
    p = X_test_const.shape[1] - 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # Convert to numpy arrays for permutation importance
    X_test_np = X_test_const.values
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test
    perm_importances = compute_permutation_importance(
        ols_model, X_test_np, y_test_np, X_test_const.columns.tolist(), n_repeats=10
    )
    perm_imp_dict = {name: (mean_imp, std_imp) for name, mean_imp, std_imp in perm_importances}
    for feat in X_train_const.columns:
        results.append({
            **meta,
            "predictors": "text_only" if meta["predictors"] == "text" else f"text+{meta['attn_method']}",
            "dependent": meta["dependent"],
            "feature_name": feat,
            "coefficient": ols_model.params[feat],
            "t": ols_model.tvalues[feat],
            "std_error": ols_model.bse[feat],
            "p_value": ols_model.pvalues[feat],
            "rsquared": r2,
            "rsquared_adj": r2_adj,
            "rmse": rmse,
            "perm_importance_mean": perm_imp_dict.get(feat, (np.nan, np.nan))[0],
            "perm_importance_std": perm_imp_dict.get(feat, (np.nan, np.nan))[1]
        })

for task in tasks:
    for model_name in llm_models:
        #text_df = pd.read_csv(f'materials/text_features_{task}_{model_name}.csv')
        text_df = pd.read_csv(f'materials/text_features_{task}_{model_name}.csv')
        text_df['role'] = text_df['role'].map({'function': 0, 'content': 1})
        X_text = text_df[['frequency', 'length', 'surprisal', 'role']]
        gaze_df, _, _ = load_processed_data(attn_method="raw", task=task, model_name=model_name)
        y_gaze = gaze_df[FEATURES]
        y_pca, _, _, _ = apply_pca(gaze_df, FEATURES, n_components=1)
        X_train_text, X_test_text, y_train_gaze, y_test_gaze, y_train_pca, y_test_pca = train_test_split(
            X_text, y_gaze, y_pca, test_size=0.2, random_state=42
        )
        #X_train_text_scaled, X_test_text_scaled = scale_text_features(X_train_text, X_test_text)

        # TextOnly Gaze (baseline) for each model
        for col in y_gaze.columns:
            run_ols(
                X_train_text, X_test_text, y_train_gaze[col], y_test_gaze[col],
                predictors=list(X_train_text.columns),
                meta={"task": task, "llm_model": model_name, "attn_method": "none", "predictors": "text", "dependent": col}
            )
        # TextOnly PCA (baseline) for each model
        run_ols(
            X_train_text, X_test_text, y_train_pca, y_test_pca,
            predictors=list(X_train_text.columns),
            meta={"task": task, "llm_model": model_name, "attn_method": "none", "predictors": "text", "dependent": "pca"}
        )

        for attn_method in attn_methods:
            # if attn_method == "flow" and model_name == "bert" and task == "task2":
            #     continue
            gaze_df, attention_tensor, save_dir = load_processed_data(attn_method=attn_method, task=task, model_name=model_name)
            print(f"Processing {task} with {model_name} and {attn_method} attention: {attention_tensor.shape}")
            y_gaze = gaze_df[FEATURES]
            y_pca, _, _, _ = apply_pca(gaze_df, FEATURES, n_components=1)
            X_attention, attn_names = get_attention_features(attention_tensor, gaze_df, model_name, attn_method)
            X_text_attn = X_text.copy()
            for idx, name in enumerate(attn_names):
                X_text_attn[name] = X_attention[:, idx]
            # Split
            X_train_attn, X_test_attn, y_train_gaze, y_test_gaze, y_train_pca, y_test_pca = train_test_split(
                X_text_attn, y_gaze, y_pca, test_size=0.2, random_state=42
            )
            #X_train_attn_scaled, X_test_attn_scaled = scale_text_features(X_train_attn, X_test_attn)
            # Add attention columns (do not scale)
            for name in attn_names:
                X_train_attn[name] = X_train_attn[name].values
                X_test_attn[name] = X_test_attn[name].values
            predictors = list(X_train_attn.columns)
            # 3/5/7. Attention Gaze
            for col in y_gaze.columns:
                run_ols(
                    X_train_attn, X_test_attn, y_train_gaze[col], y_test_gaze[col],
                    predictors=predictors,
                    meta={"task": task, "llm_model": model_name, "attn_method": attn_method, "predictors": f"text+{attn_method}", "dependent": col}
                )
            # 4/6/8. Attention PCA
            run_ols(
                X_train_attn, X_test_attn, y_train_pca, y_test_pca,
                predictors=predictors,
                meta={"task": task, "llm_model": model_name, "attn_method": attn_method, "predictors": f"text+{attn_method}", "dependent": "pca"}
            )

# Save unified results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/ols_unified_performance.csv", index=False)
print("Unified OLS results saved to outputs/ols_unified_performance.csv")