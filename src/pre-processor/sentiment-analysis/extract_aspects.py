#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Author: Arman Kabiri
### Email: Arman.Kabiri94@gmail.com
### Date: Oct. 6, 2020


# In[ ]:


# In this script, we extract all aspects of the products fallen in a specified category.


# In[6]:


import pandas as pd
import random
import os
import json


# In[3]:


df = pd.read_json('data/overnight.jl',lines=True)


# In[97]:


REQUESTED_CATEGORY = 'food'
data_dir = 'data'


# In[98]:


sub_categories = {}
with open (os.path.join(data_dir,'categories_'+REQUESTED_CATEGORY+'.csv')) as file:
    for line in file:
        cat,_ = line.split('\t')
        sub_categories[cat] = set()


# In[99]:


for entry in df[['c','adjs']].iloc():
    if entry['c'] in sub_categories:
        sub_categories[entry['c']].update([list(i.keys())[0] for i in entry['adjs']])


# In[100]:


class JsonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# In[101]:


with open(os.path.join(data_dir,'aspects_'+REQUESTED_CATEGORY+'.json'),'w') as file:
    json_object = json.dump(sub_categories, indent = 4, ensure_ascii=False, cls = JsonSetEncoder, fp=file)

