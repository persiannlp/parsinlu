#!/usr/bin/env python
# coding: utf-8

# This script is used to extract reviews based on the sub-categories of the requested category.

import pandas as pd
import random
import tqdm

df = pd.read_json('data/overnight.jl',lines=True)

print(f"# products = {df.shape[0]}")

REQUESTED_CATEGORY = 'food'

requested_categories_file = f'data/categories_{REQUESTED_CATEGORY}.csv'
categories = dict()

with open(requested_categories_file,'r') as file:
    for line in file:
        cat,n = line.split('\t')
        categories[cat.strip()] = 0

max_num_of_reviews=1500
max_num_of_reviews_each_subcat = max_num_of_reviews/len(categories)
min_tokens=10

reviews = list()

for entry in df.iloc():
    cmts = entry['cmts']
    category = entry['c']
    
    if category in categories and categories[category] < max_num_of_reviews_each_subcat:
        i=0
        
        while i<len(cmts) and categories[category] < max_num_of_reviews_each_subcat:
            comment = cmts[i]
            if len(comment['txt'].split())>min_tokens :
                reviews.append((comment['txt'].replace('\n',' '), comment['pol'] if comment['pol'] is not None else '-', category))
                categories[category] += 1
            i+=1

with open(f'data/reviews_{REQUESTED_CATEGORY}_{max_num_of_reviews}.csv','w') as file:
    for review in reviews:
        file.write(review[0]+'\t'+str(review[1])+'\t'+review[2]+'\n')