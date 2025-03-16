import pandas as pd
from scripts.data_processing.utils_ZuCo import * 

# Task 2
datatransform_t2_word = DataTransformer('task2', level='word', scaling='min-max', fillna='zeros')
# datatransform_t2_sent = DataTransformer('task2', level='sentence', scaling='min-max', fillna='zeros')
# Task 3
# datatransform_t3_word = DataTransformer('task3', level='word', scaling='min-max', fillna='zeros')
# datatransform_t3_sent = DataTransformer('task3', level='sentence', scaling='min-max', fillna='zeros')

sbjs_t2_word = []
for i in range(12):
    df = datatransform_t2_word(i)
    df['participantID'] = i  # Add a new column for participant number
    sbjs_t2_word.append(df)

# show the first couple of rows for subject 1 for task 2
#print(sbjs_t2_word[0].head())

#print(sbjs_t2_sent[0].head())
# convert DataFrame to .csv file and save it to path 
combined_df = pd.concat(sbjs_t2_word, ignore_index=True)
combined_df.to_csv('data/task2/processed/all_participants.csv', index=False)