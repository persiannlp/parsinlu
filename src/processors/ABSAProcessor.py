#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import json
from transformers import InputExample, DataProcessor


# In[39]:


class ABSAProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        self.labels = ['-3','-2','-1','0','1','2','3']
        
    def load_data_jsonl(self, data_path):
        with open(data_path, 'r') as file:
            lines = file.readlines()

        dataset = []

        for line in lines:
            data = json.loads(line.replace('\'', '\"'))
            dataset.append(data)  
            
        return dataset

    def get_train_examples(self, data_dir):
        """See base class."""
        
        dataset = self.load_data_jsonl(os.path.join(data_dir,"food_train.jsonl"))
        examples = []
        
        for i,entry in enumerate(dataset):
            
            guid = f'train-r{entry["review_id"]}-e{entry["example_id"]}'
            text_a = entry["review"]
            text_b = entry["question"]
            label = str(entry["label"])
            
            if label not in self.labels:
                continue
            
            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            
#             if i < 10:
#                 print(example)
            
            examples.append(example)
        
        return examples
 

    def get_dev_examples(self, data_dir):
        """See base class."""
        
        dataset = self.load_data_jsonl(os.path.join(data_dir,"food_dev.jsonl"))
        
        examples = []
        
        for i,entry in enumerate(dataset):
            
            guid = f'dev-r{entry["review_id"]}-e{entry["example_id"]}'
            text_a = entry["review"]
            text_b = entry["question"]
            label = str(entry["label"])
            
            if label not in self.labels:
                continue
            
            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            
#             if i < 10:
#                 print(example)
            
            examples.append(example)
        
        return examples
    
    def get_test_examples(self, data_dir):
        """See base class."""
        
        dataset = self.load_data_jsonl(os.path.join(data_dir,"food_test.jsonl"))
        
        examples = []
        
        for i,entry in enumerate(dataset):
            
            guid = f'test-r{entry["review_id"]}-e{entry["example_id"]}'
            text_a = entry["review"]
            text_b = entry["question"]
            label = str(entry["label"])
            
            if label not in self.labels:
                continue
            
            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            
#             if i < 10:
#                 print(example)
            
            examples.append(example)
        
        return examples

    def get_labels(self):
        """See base class."""
        return self.labels


# In[43]:


# train = ABSAProcessor().get_dev_examples('../../data/sentiment-analysis/')


# In[44]:


# train[0:5]


# In[ ]:




