import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from scripts.probing.raw import subset

# Load human data
human_df = pd.read_csv('data/task2/processed/averaged_participants.csv')
human_df = human_df[human_df['Sent_ID'] < subset]  # Subset for testing

model_data = torch.load("/scratch/7982399/thesis/outputs/attention_processed.pt")  
attention = model_data['attention_processed']

# Eye-trackingFeatures to correlate
features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'WordLen']

num_layers, num_sentences, max_seq_len = attention.shape
# This maps each token in human_df to a flattened attention index
token_indices = []
for sent_id in human_df['Sent_ID'].unique():
    sent_tokens = human_df[human_df['Sent_ID'] == sent_id]
    for _, token_row in sent_tokens.iterrows():
        # Flattened index = sentence_index * max_seq_len + word_position_in_sentence
        flattened_idx = sent_id * max_seq_len + token_row['Word_ID']
        token_indices.append(flattened_idx)

token_indices = np.array(token_indices)  # shape [num_real_tokens]

# Flatten attention tensor across sentence and token dimensions
attention_flat = attention.reshape(num_layers, -1)  # [32, 5 * 37]

# Extract non-padded token values only
attention_real_tokens = attention_flat[:, token_indices]  # [layers, num_real_tokens]

# Extract feature matrix from human data
feature_matrix = human_df[features].to_numpy()  # [num_real_tokens, num_features]

# --- Correlation Analysis ---
results = []

for layer_idx in range(num_layers):
    attention_values = attention_real_tokens[layer_idx].numpy()  # [num_real_tokens]
    for feature_idx, feature_name in enumerate(features):
        human_feature_values = feature_matrix[:, feature_idx]
        
        # Only compute if valid
        if len(attention_values) > 1:
            r_value, p_value = pearsonr(attention_values, human_feature_values)
        else:
            r_value, p_value = (np.nan, np.nan)

        results.append({
            'layer': layer_idx,
            'feature': feature_name,
            'pearson_r': r_value,
            'p_value': p_value
        })

results_df = pd.DataFrame(results)
print(results_df)