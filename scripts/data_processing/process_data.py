import pandas as pd
import math
import json

# Script for further processing of participant data
# Steps:
# Align Sent_IDs to be global across participants
# Remove '_NR' from Sent_IDs for convenience of sorting the df
# Extract sentences directly from participant data to ensure perfect alignment
# Average participant data for each sentence and word
task = "task3"
# Load CSV
df = pd.read_csv(f'data/{task}/processed/all_participants.csv')


def adjust_sent_id(missing_start, missing_end, sent_id):
    pos, suffix = sent_id.split('_')
    pos = int(pos)
    if pos >= missing_start - 1:
        pos += missing_end - missing_start + 1
    return f"{pos}_{suffix}"

# Missing sentences:
# p6 task 2: 1 - 50
# p11 task 2: 51 - 100
# p3 task 3: 179 - 225
# p7 task 3: 360 - 407 AND by observation: 317 - 364 #316, 363 kinda works  #shift is 47 prob true
# p11 task 3: 271 - 314
# p11 task 3: 363 - 407

print(df[(df['participantID'] == 7) & 
                 (df['Sent_ID'].str.split('_').str[0].astype(int) >= 268) & 
                 (df['Sent_ID'].str.split('_').str[0].astype(int) <= 315)]
)

if task == "task2":
    df.loc[df['participantID'] == 6, 'Sent_ID'] = df.loc[df['participantID'] == 6, 'Sent_ID'].map(lambda s: adjust_sent_id(1, 50, s))
    df.loc[df['participantID'] == 11, 'Sent_ID'] = df.loc[df['participantID'] == 11, 'Sent_ID'].map(lambda s: adjust_sent_id(51, 100, s))
elif task == "task3":
    df.loc[df['participantID'] == 3, 'Sent_ID'] = df.loc[df['participantID'] == 3, 'Sent_ID'].map(lambda s: adjust_sent_id(179, 225, s))
    df.loc[df['participantID'] == 7, 'Sent_ID'] = df.loc[df['participantID'] == 7, 'Sent_ID'].map(lambda s: adjust_sent_id(360, 407, s))
    df.loc[df['participantID'] == 11, 'Sent_ID'] = df.loc[df['participantID'] == 11, 'Sent_ID'].map(lambda s: adjust_sent_id(271, 314, s))
    df.loc[df['participantID'] == 11, 'Sent_ID'] = df.loc[df['participantID'] == 11, 'Sent_ID'].map(lambda s: adjust_sent_id(363, 407, s))

print(df[(df['participantID'] == 7) & 
                 (df['Sent_ID'].str.split('_').str[0].astype(int) >= 267) &
                 (df['Sent_ID'].str.split('_').str[0].astype(int) <= 314)]
)
# Remove '_NR' or '_TSR' and convert Sent_ID to int
df['Sent_ID'] = df['Sent_ID'].apply(lambda x: int(x.split('_')[0]))

# Sort by Sent_ID (optional but recommended if you want them ordered)
df = df.sort_values(by=['Sent_ID', 'Word_ID'])

# Define eye-tracking features
features = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']

# Group by Sent_ID and participantID to calculate sentence-level sums
sentence_sums = df.groupby(['Sent_ID', 'participantID'])[features].transform('sum')

# Normalize each feature by dividing by the sentence-level sum
for feature in features:
    df[feature] = df[feature] / sentence_sums[feature]

# Get unique combinations of Sent_ID and Word_ID with their corresponding words
sentence_items = df[['Sent_ID', 'Word_ID', 'Word']].drop_duplicates()

# Group by Sent_ID to get list of words for each sentence
sentences = sentence_items.groupby('Sent_ID')['Word'].apply(list).tolist()

# Clean nan values
sentences = [[word for word in sentence if pd.notnull(word)] for sentence in sentences]

# Save the cleaned sentences to a JSON file
with open(f'materials/sentences_{task}.json', 'w') as f:
    json.dump(sentences, f, indent=4)

# Group by Sent_ID and Word_ID to average across participants
agg_df = df.groupby(['Sent_ID', 'Word_ID', 'Word'])[features].mean().reset_index()

agg_df.to_csv(f'data/{task}/processed/processed_participants.csv', index=False)