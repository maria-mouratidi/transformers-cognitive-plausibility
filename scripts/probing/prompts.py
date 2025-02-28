prompt_task2 = """

You will be presented with sentences that contain semantic relations. 
Your task is to process these sentences as some sentences will be followed by a multiple-choice question about their content. 
These questions serve as a control condition and will require you to select the correct answer from a set of options.  

- After reading each sentence, type 1 to continue.  
- When a multiple-choice question appears, select the correct option by typing the corresponding number.  

"""

# Procedure: Pass the instructions as a prompt, and all the sentences along with the control questions at once.

prompt_task2 = """ 

You will be presented with sentences that contain specific types of semantic relations.
 Your task is to focus on identifying a particular relation type in each block of sentences.
The relation types include: **award**, **education**, **employer**, **founder**, **job_title**, **nationality**, **political_affiliation**, **visited**, and **wife**.  

Each block will only contain sentences related to one relation type.
Before the block starts, you will receive a definition of the relation type, along with three sample sentences as a practice round.
Use this information to guide your judgments during the task.  

For each sentence:  
- Carefully read the sentence.  
- After reading, answer the following question:  
**"Does this sentence contain the [relation_type] relation?"**  
- Respond with **Yes** or **No** based on whether the sentence expresses the specified relation.  

"""