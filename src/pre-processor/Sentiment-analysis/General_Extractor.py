#!/usr/bin/env python
# coding: utf-8

# In[3]:


### Author: Arman Kabiri
### Email: Arman.Kabiri94@gmail.com
### Date: Aug. 11, 2020


# In[1]:


# This is a general script for extracting all crawled reviews.


# In[1]:


import pandas as pd
import random


# In[3]:


df = pd.read_json('data/overnight.jl',lines=True)


# In[11]:


df['cmts'][0][0]


# In[6]:


df['adjs'][0]


# ### Size:

# In[3]:


print(f"# products = {df.shape[0]}")


# In[4]:


comments_size = 0
comments_list = df['cmts']

for comments in comments_list:
    comments_size += len(comments)


# In[5]:


print(f"# Comments = {comments_size}")


# ### Samples:

# df.loc[4]['cmts']

# print(df.loc[2].cmts[7]['txt'])

# df.loc[0]['adjs']

# ### Extracting Categories:

# In[6]:


categories = set(df['c'])
print(categories)


# ### Extracting All Comments:

# In[13]:


comments_list = list()
for row in df.iloc:
    category = row['c'] if row['c'] is not None else '-'
    for comment in row['cmts']:
        comments_list.append((comment['txt'].replace('\n',' '), comment['pol'] if comment['pol'] is not None else '-', category))


# In[14]:


comments_list[122]


# In[15]:


rand_indices = set()
with open('sample_digikala_comments.txt','w') as file:
    while True:
        index = random.randint(0,len(comments_list))
        if index in rand_indices:
            continue
        else:
            rand_indices.add(index)
            file.write(comments_list[index][0]+'\t'+str(comments_list[index][1])+'\t'+comments_list[index][2])
            file.write('\n')
        if len(rand_indices)>=100:
            break


# ### Extracting Categories

# In[7]:


categories_dict = dict()
for row in df.iloc:
    category = row['c'] if row['c'] is not None else '-'
    if category not in categories_dict:
        categories_dict[category] = 0
    categories_dict[category]+=len(row['cmts'])


# In[8]:


categories_dict


# In[9]:


with open('categories.csv','w') as file:
    for cat,count in categories_dict.items():
        file.write(cat.replace('\t',' ')+"\t"+str(count)+"\n")


# In[ ]:




