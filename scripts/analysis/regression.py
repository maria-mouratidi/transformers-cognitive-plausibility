import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, wilcoxon
from scripts.analysis.correlation import load_processed_data, map_token_indices
from scripts.analysis.correlation_pca import apply_pca

# ------------------ Load and Prepare Data ------------------

attn_method, task = "raw", "task3"
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
scaler_X = StandardScaler()
X_train_text_scaled = scaler_X.fit_transform(X_train_text)
X_test_text_scaled = scaler_X.transform(X_test_text)

scaler_attn = StandardScaler()
X_train_attn_scaled = scaler_attn.fit_transform(X_train_attn)
X_test_attn_scaled = scaler_attn.transform(X_test_attn)

# ------------------ Train Models ------------------
text_only_gaze_model = LinearRegression()
text_only_gaze_model.fit(X_train_text_scaled, y_train_gaze)
preds_text_only_gaze = text_only_gaze_model.predict(X_test_text_scaled)

text_only_pca_model = LinearRegression()
text_only_pca_model.fit(X_train_text_scaled, y_train_pca)
preds_text_only_pca = text_only_pca_model.predict(X_test_text_scaled)
preds_text_only_pca_inv = pca_obj.inverse_transform(preds_text_only_pca)

text_attn_pca_model = LinearRegression()
text_attn_pca_model.fit(X_train_attn_scaled, y_train_pca)
preds_text_attn_pca = text_attn_pca_model.predict(X_test_attn_scaled)
preds_text_attn_pca_inv = pca_obj.inverse_transform(preds_text_attn_pca)

text_attn_gaze_model = LinearRegression()
text_attn_gaze_model.fit(X_train_attn_scaled, y_train_gaze)
preds_text_attn_gaze = text_attn_gaze_model.predict(X_test_attn_scaled)

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

# ------------------ Save Results ------------------
output_file = f"outputs/{task}/{attn_method}/regression_results.txt"
with open(output_file, "w") as f:
    f.write("Regression Results\n==================\n\n")

    f.write("TextOnly_GazeModel vs TextOnly_PCAModel (inverted)\n")
    f.write(f"t-test: t={t_stat_1:.4f}, p={p_val_1:.4f}\nWilcoxon: stat={w_stat_1:.4f}, p={w_p_1:.4f}\n")
    f.write(f"TextOnly_GazeModel: mean={np.mean(errors_text_only_gaze):.4f}, std={np.std(errors_text_only_gaze):.4f}\n")
    f.write(f"TextOnly_PCAModel (inv): mean={np.mean(errors_text_only_pca_inv):.4f}, std={np.std(errors_text_only_pca_inv):.4f}\n\n")

    f.write("TextOnly_PCAModel vs TextAttn_PCAModel\n")
    f.write(f"t-test: t={t_stat_2:.4f}, p={p_val_2:.4f}\nWilcoxon: stat={w_stat_2:.4f}, p={w_p_2:.4f}\n")
    f.write(f"TextOnly_PCAModel: mean={np.mean(errors_text_only_pca):.4f}, std={np.std(errors_text_only_pca):.4f}\n")
    f.write(f"TextAttn_PCAModel: mean={np.mean(errors_text_attn_pca):.4f}, std={np.std(errors_text_attn_pca):.4f}\n\n")

    f.write("TextAttn_GazeModel vs TextAttn_PCAModel (inverted)\n")
    f.write(f"t-test: t={t_stat_3:.4f}, p={p_val_3:.4f}\nWilcoxon: stat={w_stat_3:.4f}, p={w_p_3:.4f}\n")
    f.write(f"TextAttn_GazeModel: mean={np.mean(errors_text_attn_gaze):.4f}, std={np.std(errors_text_attn_gaze):.4f}\n")
    f.write(f"TextAttn_PCAModel (inv): mean={np.mean(errors_text_attn_pca_inv):.4f}, std={np.std(errors_text_attn_pca_inv):.4f}\n\n")

    f.write("Baseline vs TextAttn_PCAModel\n")
    f.write(f"t-test: t={t_stat_4:.4f}, p={p_val_4:.4f}\nWilcoxon: stat={w_stat_4:.4f}, p={w_p_4:.4f}\n")
    f.write(f"Baseline: mean={np.mean(errors_baseline):.4f}, std={np.std(errors_baseline):.4f}\n")
    f.write(f"TextAttn_PCAModel: mean={np.mean(errors_text_attn_pca):.4f}, std={np.std(errors_text_attn_pca):.4f}\n\n")

print(f"Results saved to {output_file}")
