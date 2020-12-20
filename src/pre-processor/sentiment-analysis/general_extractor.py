#!/usr/bin/env python
# coding: utf-8

# This is a general script for extracting all crawled reviews.

import pandas as pd
import random

df = pd.read_json('data/overnight.jl',lines=True)

print(f"# products = {df.shape[0]}")

comments_size = 0
comments_list = df['cmts']

for comments in comments_list:
    comments_size += len(comments)

print(f"# Comments = {comments_size}")

#### Extracting Categories:
categories = set(df['c'])
print(categories)


#### Extracting All Comments:
comments_list = list()
for row in df.iloc:
    category = row['c'] if row['c'] is not None else '-'
    for comment in row['cmts']:
        comments_list.append((comment['txt'].replace('\n',' '), comment['pol'] if comment['pol'] is not None else '-', category))

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

#### Extracting Categories
categories_dict = dict()
for row in df.iloc:
    category = row['c'] if row['c'] is not None else '-'
    if category not in categories_dict:
        categories_dict[category] = 0
    categories_dict[category]+=len(row['cmts'])

    
with open('categories.csv','w') as file:
    for cat,count in categories_dict.items():
        file.write(cat.replace('\t',' ')+"\t"+str(count)+"\n")