#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import os
import json

df = pd.read_json('data/overnight.jl',lines=True)

REQUESTED_CATEGORY = 'food'
data_dir = 'data'

sub_categories = {}
with open (os.path.join(data_dir,'categories_'+REQUESTED_CATEGORY+'.csv')) as file:
    for line in file:
        cat,_ = line.split('\t')
        sub_categories[cat] = set()

for entry in df[['c','adjs']].iloc():
    if entry['c'] in sub_categories:
        sub_categories[entry['c']].update([list(i.keys())[0] for i in entry['adjs']])


class JsonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

    
with open(os.path.join(data_dir,'aspects_'+REQUESTED_CATEGORY+'.json'),'w') as file:
    json_object = json.dump(sub_categories, indent = 4, ensure_ascii=False, cls = JsonSetEncoder, fp=file)