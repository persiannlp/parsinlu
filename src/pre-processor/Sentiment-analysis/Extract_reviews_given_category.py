#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Author: Arman Kabiri
### Email: Arman.Kabiri94@gmail.com
### Date: Oct. 5, 2020


# In[ ]:


# This script extract crawled reviews for the requested category. We have used this to extract reviews for Food category.


# In[2]:


import pandas as pd
import random


# In[3]:


df = pd.read_json('overnight.jl',lines=True)


# In[4]:


print(f"# products = {df.shape[0]}")


# In[68]:


REQUESTED_CATEGORY = 'food'


# ### Reading Sub-Categories

# In[60]:


requested_categories_file = f'categories_{REQUESTED_CATEGORY}.txt'
categories = dict()


# In[61]:


with open(requested_categories_file,'r') as file:
    for line in file:
        categories[line.strip()] = 0


# ### Extracting Reviews

# In[63]:


reviews = list()
for key in categories: categories[key] = 0
for entry in df.iloc():
    cmts = entry['cmts']
    category = entry['c']
    if category in categories:
        for comment in cmts:
            reviews.append((comment['txt'].replace('\n',' '), comment['pol'] if comment['pol'] is not None else '-', category))
            categories[category] += 1
            
##Sorting The categories:
categories = {k: v for k, v in sorted(categories.items(), key=lambda item: item[1],reverse=True)}


# ### Writing Reviews To File:

# In[66]:


with open(f'reviews_{REQUESTED_CATEGORY}.csv','w') as file:
    for review in reviews:
        file.write(review[0]+'\t'+str(review[1])+'\t'+review[2]+'\n')


# In[67]:


with open(f'categories_{REQUESTED_CATEGORY}.csv','w') as file:
    for cat,count in categories.items():
        if count > 0:
            file.write(cat.replace('\t',' ').strip()+"\t"+str(count)+"\n")

