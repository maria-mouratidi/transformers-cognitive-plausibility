import pandas as pd
from scripts.data_processing.utils_ZuCo import * 

# Task 2
datatransform_t2 = DataTransformer('task2', level='word', scaling='raw', fillna='zeros')
# Task 3
# datatransform_t3 = DataTransformer('task3', level='word', scaling='raw', fillna='zeros')

sbjs_t2 = []
for i in range(12):
    df = datatransform_t2(i)
    df['participantID'] = i  # Add a new column for participant number
    sbjs_t2.append(df)

combined_df = pd.concat(sbjs_t2, ignore_index=True)
combined_df.to_csv('data/task2/processed/all_participants.csv', index=False)