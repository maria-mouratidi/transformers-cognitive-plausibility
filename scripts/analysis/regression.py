import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import ttest_rel, wilcoxon
from scripts.analysis.correlation import load_processed_data, map_token_indices
from scripts.analysis.correlation_pca import apply_pca
from ..visuals.added_var_plot import partial_reg_plot
from scripts.analysis.perm_feat_imp import compute_permutation_importance

# ------------------ Load and Prepare Data ------------------

attn_method, task = "raw", "task2"
text_df = pd.read_csv(f'materials/text_features_{task}.csv')
text_df['role'] = text_df['role'].map({'function': 0, 'content': 1})
gaze_df, attention_tensor = load_processed_data(attn_method=attn_method, task=task)

X_text = text_df[['frequency', 'length', 'surprisal', 'role']]
gaze_features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']
y_gaze = gaze_df[gaze_features]

y_pca, pca_obj, explained_var, cum_var = apply_pca(gaze_df, gaze_features)

# ------------------ Attention Features ------------------
sent_idx, word_idx = zip(*map_token_indices(gaze_df))
sent_idx = np.array(sent_idx)
word_idx = np.array(word_idx)

selected_layers = [31]
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

# ------------------ Feature Scaling ------------------
# Create a log transformer that applies log1p (log(1+x))
log_transformer = FunctionTransformer(np.log1p, validate=True)

X_train_text_log = log_transformer.fit_transform(X_train_text)
X_test_text_log = log_transformer.transform(X_test_text)

X_train_attn_log = log_transformer.fit_transform(X_train_attn)
X_test_attn_log = log_transformer.transform(X_test_attn)

# ------------------ Train Models ------------------
text_only_gaze_model = LinearRegression()
text_only_gaze_model.fit(X_train_text_log, y_train_gaze)
preds_text_only_gaze = text_only_gaze_model.predict(X_test_text_log)

text_only_pca_model = LinearRegression()
text_only_pca_model.fit(X_train_text_log, y_train_pca)
preds_text_only_pca = text_only_pca_model.predict(X_test_text_log)
preds_text_only_pca_inv = pca_obj.inverse_transform(preds_text_only_pca)

text_attn_pca_model = LinearRegression()
text_attn_pca_model.fit(X_train_attn_log, y_train_pca)
preds_text_attn_pca = text_attn_pca_model.predict(X_test_attn_log)
preds_text_attn_pca_inv = pca_obj.inverse_transform(preds_text_attn_pca)

text_attn_gaze_model = LinearRegression()
text_attn_gaze_model.fit(X_train_attn_log, y_train_gaze)
preds_text_attn_gaze = text_attn_gaze_model.predict(X_test_attn_log)

# ------------------ Baseline ------------------
baseline_mean = y_train_pca.mean(axis=0)
baseline_preds = np.tile(baseline_mean, (y_test_pca.shape[0], 1))

# ------------------ Error Calculation ------------------
errors_text_only_gaze = np.mean((preds_text_only_gaze - y_test_gaze.values)**2, axis=1)
errors_text_only_pca = np.mean((preds_text_only_pca - y_test_pca.values)**2, axis=1)
errors_text_only_pca_inv = np.mean((preds_text_only_pca_inv - y_test_gaze.values)**2, axis=1)

errors_text_attn_pca = np.mean((preds_text_attn_pca - y_test_pca.values)**2, axis=1)
errors_text_attn_pca_inv = np.mean((preds_text_attn_pca_inv - y_test_gaze.values)**2, axis=1)
errors_text_attn_gaze = np.mean((preds_text_attn_gaze - y_test_gaze.values)**2, axis=1)
errors_baseline = np.mean((baseline_preds - y_test_pca.values)**2, axis=1)

# ------------------ Statistical Tests ------------------
t_stat_1, p_val_1 = ttest_rel(errors_text_only_gaze, errors_text_only_pca_inv)
t_stat_2, p_val_2 = ttest_rel(errors_text_only_pca, errors_text_attn_pca)
t_stat_3, p_val_3 = ttest_rel(errors_text_attn_gaze, errors_text_attn_pca_inv)
t_stat_4, p_val_4 = ttest_rel(errors_baseline, errors_text_attn_pca)

w_stat_1, w_p_1 = wilcoxon(errors_text_only_gaze, errors_text_only_pca_inv)
w_stat_2, w_p_2 = wilcoxon(errors_text_only_pca, errors_text_attn_pca)
w_stat_3, w_p_3 = wilcoxon(errors_text_attn_gaze, errors_text_attn_pca_inv)
w_stat_4, w_p_4 = wilcoxon(errors_baseline, errors_text_attn_pca)

# ------------------ Metrics Calculation ------------------
from sklearn.metrics import r2_score
n_gaze = y_test_gaze.shape[0]
n_pca = y_test_pca.shape[0]
p_text = X_test_text_log.shape[1]
p_attn = X_test_attn_log.shape[1]

r2_text_only_gaze = r2_score(y_test_gaze, preds_text_only_gaze)
rmse_text_only_gaze = np.sqrt(mean_squared_error(y_test_gaze, preds_text_only_gaze))
adj_r2_text_only_gaze = 1 - (1 - r2_text_only_gaze) * (n_gaze - 1) / (n_gaze - p_text - 1)

r2_text_only_pca = r2_score(y_test_pca, preds_text_only_pca)
rmse_text_only_pca = np.sqrt(mean_squared_error(y_test_pca, preds_text_only_pca))
adj_r2_text_only_pca = 1 - (1 - r2_text_only_pca) * (n_pca - 1) / (n_pca - p_text - 1)

r2_text_only_pca_inv = r2_score(y_test_gaze, preds_text_only_pca_inv)
rmse_text_only_pca_inv = np.sqrt(mean_squared_error(y_test_gaze, preds_text_only_pca_inv))
adj_r2_text_only_pca_inv = 1 - (1 - r2_text_only_pca_inv) * (n_gaze - 1) / (n_gaze - p_text - 1)

r2_text_attn_pca = r2_score(y_test_pca, preds_text_attn_pca)
rmse_text_attn_pca = np.sqrt(mean_squared_error(y_test_pca, preds_text_attn_pca))
adj_r2_text_attn_pca = 1 - (1 - r2_text_attn_pca) * (n_pca - 1) / (n_pca - p_attn - 1)

r2_text_attn_pca_inv = r2_score(y_test_gaze, preds_text_attn_pca_inv)
rmse_text_attn_pca_inv = np.sqrt(mean_squared_error(y_test_gaze, preds_text_attn_pca_inv))
adj_r2_text_attn_pca_inv = 1 - (1 - r2_text_attn_pca_inv) * (n_gaze - 1) / (n_gaze - p_attn - 1)

r2_text_attn_gaze = r2_score(y_test_gaze, preds_text_attn_gaze)
rmse_text_attn_gaze = np.sqrt(mean_squared_error(y_test_gaze, preds_text_attn_gaze))
adj_r2_text_attn_gaze = 1 - (1 - r2_text_attn_gaze) * (n_gaze - 1) / (n_gaze - p_attn - 1)

# For baseline (using p=1 for a constant predictor)
r2_baseline = r2_score(y_test_pca, baseline_preds)
rmse_baseline = np.sqrt(mean_squared_error(y_test_pca, baseline_preds))
adj_r2_baseline = 1 - (1 - r2_baseline) * (n_pca - 1) / (n_pca - 1 - 1)


# ------------------ Feature importance explainability measures ------------------
partial_reg_plot(
    X=X_train_attn_log,
    y=y_train_pca,
    feature_names=list(X_text_attn.columns),
    attention_feature=f"attention_layer_{selected_layers[0]}",
    task=task,
    attn_method=attn_method
)

# Compute permutation importances on test set
importances = compute_permutation_importance(
    model=text_attn_pca_model,
    X=X_test_attn_log,
    y=y_test_pca,
    feature_names=list(X_text_attn.columns),
    metric_func=lambda y_true, y_pred: mean_squared_error(y_true, y_pred, multioutput='uniform_average'),
    n_repeats=20
)

# Sort importances in descending order by importance score
sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
# ------------------ Save Results ------------------
output_file = f"outputs/{task}/{attn_method}/regression_results.txt"
with open(output_file, "w") as f:
    f.write(f"Regression Results {task} {attn_method}\n==================\n\n")
    
    f.write("TextOnly_GazeModel vs TextOnly_PCAModel (inverted)\n")
    f.write(f"t-test: t={t_stat_1:.4f}, p={p_val_1:.4f}\n")
    f.write(f"Wilcoxon: stat={w_stat_1:.4f}, p={w_p_1:.4f}\n")
    f.write("TextOnly_GazeModel Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_only_gaze:.4f}\n")
    f.write(f"  RMSE: {rmse_text_only_gaze:.4f}\n")
    f.write("TextOnly_PCAModel (inverted) Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_only_pca_inv:.4f}\n")
    f.write(f"  RMSE: {rmse_text_only_pca_inv:.4f}\n\n")
    
    f.write("TextOnly_PCAModel vs TextAttn_PCAModel\n")
    f.write(f"t-test: t={t_stat_2:.4f}, p={p_val_2:.4f}\n")
    f.write(f"Wilcoxon: stat={w_stat_2:.4f}, p={w_p_2:.4f}\n")
    f.write("TextOnly_PCAModel Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_only_pca:.4f}\n")
    f.write(f"  RMSE: {rmse_text_only_pca:.4f}\n")
    f.write("TextAttn_PCAModel Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_attn_pca:.4f}\n")
    f.write(f"  RMSE: {rmse_text_attn_pca:.4f}\n\n")
    
    f.write("TextAttn_GazeModel vs TextAttn_PCAModel (inverted)\n")
    f.write(f"t-test: t={t_stat_3:.4f}, p={p_val_3:.4f}\n")
    f.write(f"Wilcoxon: stat={w_stat_3:.4f}, p={w_p_3:.4f}\n")
    f.write("TextAttn_GazeModel Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_attn_gaze:.4f}\n")
    f.write(f"  RMSE: {rmse_text_attn_gaze:.4f}\n")
    f.write("TextAttn_PCAModel (inverted) Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_attn_pca_inv:.4f}\n")
    f.write(f"  RMSE: {rmse_text_attn_pca_inv:.4f}\n\n")
    
    f.write("Baseline vs TextAttn_PCAModel\n")
    f.write(f"t-test: t={t_stat_4:.4f}, p={p_val_4:.4f}\n")
    f.write(f"Wilcoxon: stat={w_stat_4:.4f}, p={w_p_4:.4f}\n")
    f.write("Baseline Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_baseline:.4f}\n")
    f.write(f"  RMSE: {rmse_baseline:.4f}\n")
    f.write("TextAttn_PCAModel Metrics:\n")
    f.write(f"  Adjusted R²: {adj_r2_text_attn_pca:.4f}\n")
    f.write(f"  RMSE: {rmse_text_attn_pca:.4f}\n\n")

    f.write("Permutation Feature Importances (higher = more important):\n")
    for name, imp, std in sorted_importances:
        f.write(f"{name}: {imp:.6f} ± {std:.6f}\n")
    
print(f"Regression results saved to {output_file}")