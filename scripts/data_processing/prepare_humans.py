import pandas as pd

# Load CSV
df = pd.read_csv('data/task2/processed/all_participants.csv')

# 1. Extract sentences as lists of strings, ensuring each sentence appears only once
# Get unique combinations of Sent_ID and Word_ID with their corresponding words
unique_words = df[['Sent_ID', 'Word_ID', 'Word']].drop_duplicates().sort_values(['Sent_ID', 'Word_ID'])

# Group by Sent_ID to get list of words for each sentence
sentences = unique_words.groupby('Sent_ID')['Word'].apply(list).tolist()

# 2. Extract features, average across participants
# Define eye-tracking features
features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'WordLen']

# Group by Sent_ID and Word_ID to average across participants
agg_df = df.groupby(['Sent_ID', 'Word_ID', 'Word'])[features].mean().reset_index()

# This will give you one row per word (per sentence), averaged across participants

# Optional: if you want to keep it sentence-level as well
grouped_features = agg_df.groupby('Sent_ID')

# Example of how to prepare data for correlation
# You could flatten this into a long list of feature values
all_feature_values = {feature: [] for feature in features}
for _, row in agg_df.iterrows():
    for feature in features:
        all_feature_values[feature].append(row[feature])

# Now all_feature_values is ready for correlation with another group's data
# For example, you can pass these lists into scipy.stats.pearsonr or pd.DataFrame.corr()

print(f"Number of sentences: {len(sentences)}")
print(f"First sentence: {sentences[0]}")  # Show first sentence as a list of words
print(pd.DataFrame(all_feature_values).head())  # Show a preview of averaged features