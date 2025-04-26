from scripts.analysis.correlation import load_processed_data, map_token_indices
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import ttest_rel

# Load text features and human (eye-gaze) data
text_features = pd.read_csv('materials/text_features.csv')

# Load attention data using the correlation routine (adjust attn_method and task as needed)
attn_method, task = "raw", "task2"
human_df, attention = load_processed_data(attn_method=attn_method, task=task)

# Convert the binary 'role' feature from string to numeric (e.g., 'function' -> 0, 'content' -> 1)
text_features['role'] = text_features['role'].map({'function': 0, 'content': 1})

# Define eye-gaze target variables
y = human_df[['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']]

# Expected shape of attention: [num_layers, num_sentences, max_seq_len]

# Get token mapping for each row in human_df from the correlation script helper
token_indices = map_token_indices(human_df)
sent_ids, word_ids = zip(*token_indices)  # each is a tuple of indices
sent_ids = np.array(sent_ids)
word_ids = np.array(word_ids)

# Define three layers: first, middle, and last
num_layers = attention.shape[0]
selected_layers = [0, num_layers // 2, num_layers - 1]
layer_names = {0: "First Layer", num_layers // 2: "Middle Layer", num_layers - 1: "Last Layer"}

for layer in selected_layers:
    print("\n==============================")
    print(f"Regression Model with Attention from {layer_names[layer]}")
    print("==============================")
    
    # Extract attention values for each sample using the token indices:
    # This assumes attention is indexed as [layer, sent_id, word_id]
    attn_values = attention[layer, sent_ids, word_ids]  # shape: (n_samples,)
    
    # Create new predictor DataFrame by adding the attention feature to the existing text features.
    # (We assume the order of rows in text_features corresponds to that in human_df)
    X = text_features[['frequency', 'length', 'surprisal', 'role']].copy()
    X['attention'] = attn_values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale X features; note that now we have an extra attention column.
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale y values with MinMaxScaler (since they represent proportions)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create and fit multi-output regression model using the combined features
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict on the test set using our model
    y_pred = model.predict(X_test_scaled)
    
    # Implement mean baseline: predict the mean of y_train (scaled) for each eye-gaze feature
    baseline = y_train_scaled.mean(axis=0)
    y_baseline = np.tile(baseline, (len(y_test_scaled), 1))
    
    # Calculate and print mean squared error for each eye-gaze feature for the regression model
    mse_metrics = {}
    for idx, col in enumerate(y.columns):
        mse_metrics[col] = mean_squared_error(y_test_scaled[:, idx], y_pred[:, idx])
    
    print("Mean Squared Errors (MSE) per feature (Attention Model):")
    for col, mse in mse_metrics.items():
        print(f"{col}: {mse:.4f}")
    avg_mse = sum(mse_metrics.values()) / len(mse_metrics)
    print(f"Average MSE (Attention Model): {avg_mse:.4f}")
    
    # Calculate baseline MSE
    mse_baseline = {}
    for idx, col in enumerate(y.columns):
        mse_baseline[col] = mean_squared_error(y_test_scaled[:, idx], y_baseline[:, idx])
    
    print("\nMean Squared Errors (MSE) per feature (Mean Baseline):")
    for col, mse in mse_baseline.items():
        print(f"{col}: {mse:.4f}")
    avg_mse_baseline = sum(mse_baseline.values()) / len(mse_baseline)
    print(f"Average MSE (Mean Baseline): {avg_mse_baseline:.4f}")
    
    # Evaluate statistical significance between model and baseline using paired t-test on squared errors
    print("\nPaired t-test comparing Attention Model vs Mean Baseline:")
    for idx, col in enumerate(y.columns):
        errors_model = (y_test_scaled[:, idx] - y_pred[:, idx]) ** 2
        errors_baseline = (y_test_scaled[:, idx] - y_baseline[:, idx]) ** 2
        t_stat, p_val = ttest_rel(errors_baseline, errors_model)
        print(f"{col}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")