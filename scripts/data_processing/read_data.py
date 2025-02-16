from thesis.helpers.utils_ZuCo import * 
# instantiate data transformer object for task 1 (corresponds to the folder
# where results for  task 1 in  specified subdir are stored) on sentence level
# with min-max scaling 

# Task 2
datatransform_t2_word = DataTransformer('task2', level='word', scaling='min-max', fillna='zeros')
datatransform_t2_sent = DataTransformer('task2', level='sentence', scaling='min-max', fillna='zeros')
# Task 3
datatransform_t3_word = DataTransformer('task3', level='word', scaling='min-max', fillna='zeros')
datatransform_t3_sent = DataTransformer('task3', level='sentence', scaling='min-max', fillna='zeros')

sbjs_t2_word = [datatransform_t2_word(i) for i in range(1)] 
sbjs_t2_sent = [datatransform_t2_sent(i) for i in range(1)] 

# show the first couple of rows for subject 1 for task 2
print(sbjs_t2_word[0].head())
print(sbjs_t2_sent[0].head())
# convert DataFrame to .csv file and save it to path 
#sbjs_t2[0].to_csv()