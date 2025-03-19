import pandas as pd
import math
import json

# Load CSV
df = pd.read_csv('data/task2/processed/all_participants.csv')

# Adjust Sent_ID for participant 6: add an offset of 50 to align with objective sentence numbers
def adjust_sent_id_p6(sent_id):
    num, suffix = sent_id.split('_')
    return f"{int(num) + 50}_{suffix}"

df.loc[df['participantID'] == 6, 'Sent_ID'] = df.loc[df['participantID'] == 6, 'Sent_ID'].apply(adjust_sent_id_p6)

# Adjust Sent_ID for participant 11: add 50 if num >= 50
def adjust_sent_id_p11(sent_id):
    num, suffix = sent_id.split('_')
    num = int(num)
    if num >= 50:
        num += 50  # shift by +50
    return f"{num}_{suffix}"

df.loc[df['participantID'] == 11, 'Sent_ID'] = df.loc[df['participantID'] == 11, 'Sent_ID'].apply(adjust_sent_id_p11)

# Get unique combinations of Sent_ID and Word_ID with their corresponding words
unique_words = df[['Sent_ID', 'Word_ID', 'Word']].drop_duplicates().sort_values(['Sent_ID', 'Word_ID'])

# Group by Sent_ID to get list of words for each sentence
sentences = unique_words.groupby('Sent_ID')['Word'].apply(list).tolist()
# Clean nan values
sentences = [[word for word in sentence if not (isinstance(word, float) and math.isnan(word))] for sentence in sentences]
# Save the cleaned sentences to a JSON file
with open('materials/sentences.json', 'w') as f:
    json.dump(sentences, f, indent=4)

# Define eye-tracking features
features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'WordLen']

# Group by Sent_ID and Word_ID to average across participants
agg_df = df.groupby(['Sent_ID', 'Word_ID', 'Word'])[features].mean().reset_index()

agg_df.to_csv('data/task2/processed/averaged_participants.csv', index=False)

# # Optional: if you want to keep it sentence-level as well
# grouped_features = agg_df.groupby('Sent_ID')
