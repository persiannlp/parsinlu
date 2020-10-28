#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install -U scikit-learn')


# In[40]:


import json
from sklearn.model_selection import train_test_split
import random


# #### Loading Data

# In[41]:


with open('food.jsonl', 'r') as file:
    lines = file.readlines()

raw_dataset = []

for line in lines:
    data = json.loads(line.replace('\'', '\"'))
    raw_dataset.append(data)


# In[42]:


raw_dataset_size = len(raw_dataset)
print(f'dataset size is {raw_dataset_size}')


# #### Giving reviews/ extracting aspects

# In[43]:


aspects_set = set()


# In[44]:


for i, example in enumerate(raw_dataset):
    example['review_id'] = i+1
    aspects = list(example['aspects'].keys())
    aspects_set.update(aspects)


# In[45]:


aspects_set


# In[46]:


raw_dataset[0]


# #### Defining Questions for aspects

# In[47]:


aspects_questions = {}


# In[48]:


for aspect in list(aspects_set):
    aspects_questions[aspect] = f'نظر شما در مورد {aspect} محصول چیست؟'


# In[49]:


general_aspect_label = 'کلی'
aspects_questions[general_aspect_label] = 'نظر شما به صورت کلی در مورد محصول چیست؟'


# In[50]:


aspects_questions


# #### Spliting Data

# In[51]:


# CONSTANTS:
split_portions = {'train':0.8,'dev':0.1,'test':0.1}
assert(sum(split_portions.values())==1)
NONE_LABEL = -3


# In[52]:


raw_dataset_size = len(raw_dataset)

split_indices = dict()
split_indices['train'] = (0,round(split_portions['train']*raw_dataset_size))

split_indices['dev'] = (
    round(split_portions['train']*raw_dataset_size),
    round((split_portions['train']+split_portions['dev'])*raw_dataset_size)
)

split_indices['test'] = (
    round((split_portions['train']+split_portions['dev'])*raw_dataset_size),raw_dataset_size
)


# In[53]:


raw_dataset_dic = {
    'train': raw_dataset[split_indices['train'][0]:split_indices['train'][1]],
    'dev': raw_dataset[split_indices['dev'][0]:split_indices['dev'][1]],
    'test': raw_dataset[split_indices['test'][0]:split_indices['test'][1]]
}


# In[54]:


assert sum([len(dataset) for dataset in raw_dataset_dic.values()]) == raw_dataset_size


# #### Creating new QA-ABSA dataset

# In[58]:


dataset_ABSA = {'train':list(),'dev':list(),'test':list()}
for dataset_name, dataset in raw_dataset_dic.items():

    for example in dataset:
        
        product_aspects = example['aspects'].keys()

        #aspect sentiments
        for i,aspect in enumerate(aspects_set):

            entry = {'review': example['review'],
                     'review_id': str(example['review_id']),
                     'example_id': str(int(i+1)),
                     'question': aspects_questions[aspect],
                     'aspect': aspect,
                     'label': str(example['aspects'][aspect] if aspect in product_aspects else NONE_LABEL)
                    }
            
            dataset_ABSA[dataset_name].append(entry)

        # overal sentiment

        entry = {'review': example['review'],
                 'review_id': str(example['review_id']),
                 'example_id': str(int(i+2)),
                 'question': aspects_questions[general_aspect_label],
                 'aspect': general_aspect_label,
                 'label': str(example['sentiment'])
                }
        
        dataset_ABSA[dataset_name].append(entry)


# #### Adding ID labeles

# In[59]:

univ_id = 1
for dataset_name, dataset in dataset_ABSA.items():
    for i, example in enumerate(dataset):
#        example['example_id'] = str(int(i+1))
        example['guid'] = f'{dataset_name}-r{example["review_id"]}-e{example["example_id"]}'
        example['univ_id'] = str(univ_id)
        univ_id += 1


# #### Saving Data

# In[60]:


for dataset_name, dataset in dataset_ABSA.items():
    
    with open(f'food_{dataset_name}.jsonl', 'w') as f:
        
        for example in dataset:
            
#             f.write("%s\n" % annotation)
            f.write(f'{example}\n')
print('DONE')


# In[ ]:




