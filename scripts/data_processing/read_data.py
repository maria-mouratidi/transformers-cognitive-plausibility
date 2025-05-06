import pandas as pd
from scripts.data_processing.utils_ZuCo import * 

task = "task3"


datatransform_task = DataTransformer(task, level='word', scaling='raw', fillna='zeros')

sbjs = []
for i in range(12):
    df = datatransform_task(i)
    df['participantID'] = i  # Add a new column for participant number
    sbjs.append(df)

combined_df = pd.concat(sbjs, ignore_index=True)
combined_df.to_csv(f'data/{task}/processed/all_participants.csv', index=False)