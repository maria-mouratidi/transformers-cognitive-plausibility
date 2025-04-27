import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scripts.analysis.correlation import load_processed_data, map_token_indices
from scripts.analysis.correlation_pca import apply_pca

# ------------------ Load and Prepare Data ------------------
# Load datasets
text_df = pd.read_csv('materials/text_features.csv')
text_df['role'] = text_df['role'].map({'function': 0, 'content': 1})

attn_method, task = "raw", "task2"
gaze_df, attention_tensor = load_processed_data(attn_method=attn_method, task=task)

# Predictor: Text features
X_text = text_df[['frequency', 'length', 'surprisal', 'role']]

# Target: Eye-gaze features (for Model a)
gaze_features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']
y_gaze = gaze_df[gaze_features]

# Apply PCA to gaze features (for Model b and c)
y_pca, pca_obj, explained_var, cum_var = apply_pca(gaze_df, gaze_features)

# ------------------ Attention Features ------------------
# Extract token mappings
sent_idx, word_idx = zip(*map_token_indices(gaze_df))
sent_idx = np.array(sent_idx)
word_idx = np.array(word_idx)

# Option: pick specific layers (early, middle, late)
selected_layers = [31]  # example: early, middle, late layers
attention_features = []
for layer in selected_layers:
    layer_attention = attention_tensor[layer, sent_idx, word_idx]  # (n_samples,)
    attention_features.append(layer_attention)

# Stack selected attention layers into features
X_attention = np.column_stack(attention_features)
attention_feature_names = [f'attention_layer_{i}' for i in selected_layers]

# Text + attention features
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

# ------------------ Feature Scaling (only X!) ------------------
scaler_X = StandardScaler()
X_train_text_scaled = scaler_X.fit_transform(X_train_text)
X_test_text_scaled = scaler_X.transform(X_test_text)

scaler_attn = StandardScaler()
X_train_attn_scaled = scaler_attn.fit_transform(X_train_attn)
X_test_attn_scaled = scaler_attn.transform(X_test_attn)

# ------------------ Train and Evaluate Models ------------------

# --- Model a: Predict all gaze features using Text Only ---
model_a = LinearRegression()
model_a.fit(X_train_text_scaled, y_train_gaze)
preds_a = model_a.predict(X_test_text_scaled)

# mse_a = {
#     feature: mean_squared_error(y_test_gaze[feature], preds_a[:, idx])
#     for idx, feature in enumerate(gaze_features)
# }
# print("Predicting Individual Gaze Features with Text Features (MSEs):")
# for feature, mse in mse_a.items():
#     print(f"{feature}: {mse:.4f}")
# avg_mse_a = np.mean(list(mse_a.values()))
# print(f"Average MSE for Model a: {avg_mse_a:.4f}\n")

# --- Model b: Predict PCA targets using Text Only ---
model_b = LinearRegression()
model_b.fit(X_train_text_scaled, y_train_pca)
preds_b = model_b.predict(X_test_text_scaled)

# mse_b = {
#     f'PC{i+1}': mean_squared_error(y_test_pca.iloc[:, i], preds_b[:, i])
#     for i in range(y_pca.shape[1])
# }
# print("Predicting PCA Components with Text Features (MSEs):")
# for pc, mse in mse_b.items():
#     print(f"{pc}: {mse:.4f}")
# avg_mse_b = np.mean(list(mse_b.values()))
# print(f"Average MSE for Model b: {avg_mse_b:.4f}\n")

# print("Predicting (Inversed) PCA Components with Text Features (MSEs):")
preds_b_inversed = pca_obj.inverse_transform(preds_b) # Invert PCA predictions to the original gaze feature space for fair comparison

# mse_b_inversed = {
#     feature: mean_squared_error(y_test_gaze[feature], preds_b_inversed[:, idx])
#     for idx, feature in enumerate(gaze_features)
# }
# for feature, mse in mse_b_inversed.items():
#     print(f"{feature}: {mse:.4f}")
# avg_mse_b_inversed = np.mean(list(mse_b_inversed.values()))
# print(f"Average MSE for (Inversed) Model b: {avg_mse_b_inversed:.4f}\n")

# --- Model c: Predict PCA targets using Text + Attention ---
model_c = LinearRegression()
model_c.fit(X_train_attn_scaled, y_train_pca)
preds_c = model_c.predict(X_test_attn_scaled)

# mse_c = {
#     f'PC{i+1}': mean_squared_error(y_test_pca.iloc[:, i], preds_c[:, i])
#     for i in range(y_pca.shape[1])
# }
# print("Predicting PCA Components with Text + Attention (MSEs):")
# for pc, mse in mse_c.items():
#     print(f"{pc}: {mse:.4f}")
# avg_mse_c = np.mean(list(mse_c.values()))
# print(f"Average MSE for Model c: {avg_mse_c:.4f}\n")

# --- Baseline Model: Predicting Mean of PCA components ---
baseline_mean = y_train_pca.mean(axis=0)
baseline_preds = np.tile(baseline_mean, (y_test_pca.shape[0], 1))

# mse_baseline = {
#     f'PC{i+1}': mean_squared_error(y_test_pca.iloc[:, i], baseline_preds[:, i])
#     for i in range(y_pca.shape[1])
# }
# print("Baseline Model - Predicting Mean of PCA Components (MSEs):")
# for pc, mse in mse_baseline.items():
#     print(f"{pc}: {mse:.4f}")
# avg_mse_baseline = np.mean(list(mse_baseline.values()))
# print(f"Average MSE for Baseline Model: {avg_mse_baseline:.4f}\n")

# ------------------ Statistical Comparisons ------------------

# ------------------ Statistical Comparisons ------------------

# Per-sample MSEs
errors_a = np.mean((preds_a - y_test_gaze.values)**2, axis=1)
errors_b = np.mean((preds_b - y_test_pca.values)**2, axis=1)
errors_c = np.mean((preds_c - y_test_pca.values)**2, axis=1)
errors_b_original = np.mean((preds_b_inversed - y_test_gaze.values)**2, axis=1)
baseline_errors = np.mean((baseline_preds - y_test_pca.values)**2, axis=1)  # Baseline model predicts the mean of PCA components

# --- Gaze Model vs PCA Model in original gaze space ---
print("Gaze Model vs PCA Model in original gaze space")
t_stat, p_value = ttest_rel(errors_a, errors_b_original)
print(f"- Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
w_stat, w_p_val = wilcoxon(errors_a, errors_b_original)
print(f"- Wilcoxon signed-rank test: statistic = {w_stat:.4f}, p-value = {w_p_val:.4f}")
print("Error statistics:")
print(f"- Model a (Gaze): mean = {np.mean(errors_a):.4f}, std = {np.std(errors_a):.4f}")
print(f"- Model b (Inversed PCA): mean = {np.mean(errors_b_original):.4f}, std = {np.std(errors_b_original):.4f}\n")

# --- PCA Model vs Attention enhanced PCA Model ---
print("PCA Model vs Attention enhanced PCA Model")
t_stat, p_value = ttest_rel(errors_b, errors_c)  # both models are in PCA space
print(f"- Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
w_stat, w_p_val = wilcoxon(errors_b, errors_c)
print(f"- Wilcoxon signed-rank test: statistic = {w_stat:.4f}, p-value = {w_p_val:.4f}")
print("Error statistics:")
print(f"- Model b (PCA): mean = {np.mean(errors_b):.4f}, std = {np.std(errors_b):.4f}")
print(f"- Model c (Text + Attention): mean = {np.mean(errors_c):.4f}, std = {np.std(errors_c):.4f}\n")

# --- Baseline vs Attention enhanced PCA Model ---
print("Baseline vs Attention enhanced PCA Model")
t_stat, p_value = ttest_rel(baseline_errors, errors_c)
print(f"- Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
w_stat, w_p_val = wilcoxon(baseline_errors, errors_c)
print(f"- Wilcoxon signed-rank test: statistic = {w_stat:.4f}, p-value = {w_p_val:.4f}")
print("Error statistics:")
print(f"- Baseline Model: mean = {np.mean(baseline_errors):.4f}, std = {np.std(baseline_errors):.4f}")
print(f"- Model c (Text + Attention): mean = {np.mean(errors_c):.4f}, std = {np.std(errors_c):.4f}\n")