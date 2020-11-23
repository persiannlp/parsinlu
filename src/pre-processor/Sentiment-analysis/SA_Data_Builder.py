import numpy as np
import json
from sklearn.model_selection import train_test_split
import nltk
import random
import os
import argparse

parser = argparse.ArgumentParser(description='SA Dataset Generator')
parser.add_argument('--domain', type=str, help='Specify the domain you want to generate data for : {food,movie}')
parser.add_argument('--input_dir', type=str, help='Specify the location of the jsonl file.')
parser.add_argument('--output_dir', type=str, help='Specify the location of the jsonl file.')
args = parser.parse_args()

# #### Loading Data
domain_name = args.domain
with open(os.path.join(args.input_dir, args.domain) + 'jsonl', 'r') as file:
    lines = file.readlines()

raw_dataset = []

for i,line in enumerate(lines):
    try:
        data = json.loads(line.replace('\'', '\"'))
        raw_dataset.append(data)
        data['review'] = data['review'].replace('\"','')
        data['review'] = data['review'].replace("\'",'')
        data['excel_id'] = domain_name+'_'+str(int(i+1))
    except:
        print("could not read line: " + str(i))

raw_dataset_size = len(raw_dataset)

aspects_set = set()
for i, example in enumerate(raw_dataset):
    aspects = list(example['aspects'].keys())
    aspects_set.update(aspects)

for entry in raw_dataset:
    if 'category' not in entry:
        print(entry['review_id'])
        raise Exception(entry['review'] + " does not have any category associated")


# #### Defining Questions for aspects

aspects_candidate_words = {
    # For Foods"
    "ارزش خرید" : "قیمت و ارزش خرید",
    "بسته بندی" : "بسته بندی و نگهداری",
    "ارسال" : "ارسال و حمل و نقل",
    "کیفیت" : "کیفیت و تازگی",
    "ارزش غذایی" : "سلامت و ارزش غذایی",
    "طعم" : "عطر، بو، و طعم",
    # For Movies:
    "صدا" : "صداگذاری و جلوه های صوتی",
    "موسیقی" : "موسیقی",
    "داستان" : "داستان، فیلمنامه، دیالوگ ها و موضوع",
    "صحنه" : "گریم، طراحی صحنه و جلوه های ویژه ی بصری",
    "کارگردانی" : "تهیه، تدوین، کارگردانی و ساخت",
    "فیلمبرداری" : "فیلمبرداری و تصویربرداری",
    "بازی" : "شخصیت پردازی، بازیگردانی و بازی بازیگران"
}

aspects_questions = {}
for aspect in list(aspects_set):
    aspects_questions[aspect] = f'نظر شما در مورد {aspects_candidate_words[aspect]} این محصول چیست؟'

general_aspect_label = 'کلی'
aspects_questions[general_aspect_label] = 'نظر شما به صورت کلی در مورد این محصول چیست؟'

# ### Spliting Data
NONE_LABEL = -3
X = []
y = []

for rec in raw_dataset:
    X.append ({k:rec[k] for k in ('review','review_id','aspects','category','excel_id') if k in rec})
    y.append(rec['sentiment'])

test_dev_size_dic = {'movie':(0.2,0.11),'food':(0.1,0.11)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_dev_size_dic[domain_name][0], random_state=12, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_dev_size_dic[domain_name][1], random_state=12, stratify=y_train)

assert len(X_train)+len(X_test)+len(X_valid)==raw_dataset_size

# ### Setting Review_ID
review_id_counter=1
for x in X_train + X_valid + X_test:
    x['review_id'] = review_id_counter
    review_id_counter+=1

# #### Putting X and y together:
raw_dataset_dic = dict()
for entry in zip(X_train,y_train):
    entry[0].update({'sentiment':entry[1]})
    
for entry in zip(X_test,y_test):
    entry[0].update({'sentiment':entry[1]})
    
for entry in zip(X_valid,y_valid):
    entry[0].update({'sentiment':entry[1]})

raw_dataset_dic = {
    'train': X_train,
    'test': X_test,
    'dev': X_valid
}

assert sum([len(dataset) for dataset in raw_dataset_dic.values()]) == raw_dataset_size

# #### Creating new QA-ABSA dataset
def gen_question(aspect,aspects_questions,example,domain_name:str):
    
    if domain_name.lower()=='food':
        question = aspects_questions[aspect].replace('محصول',example['category'])
        
    elif domain_name.lower()=='movie':
        question = aspects_questions[aspect].replace('این محصول','فیلم ' + example['category'])

    else:
        raise Exception('Domain is not supported')
        
    return question

dataset_ABSA = {'train':list(),'dev':list(),'test':list()}
for dataset_name, dataset in raw_dataset_dic.items():

    for example in dataset:
        
        product_aspects = example['aspects'].keys()

        #aspect sentiments
        i=1
        for aspect in aspects_set:
            entry = {'review': example['review'],
                     'review_id': str(example['review_id']),
                     'example_id': str(i),
                     'excel_id':example['excel_id'],
                     'question': gen_question(aspect,aspects_questions,example,domain_name),
                     'category': example['category'],
                     'aspect': aspect,
                     'label': str(example['aspects'][aspect] if aspect in product_aspects else NONE_LABEL)
                    }
            
            dataset_ABSA[dataset_name].append(entry)
            i+=1

        # overal sentiment
        entry = {'review': example['review'],
                 'review_id': str(example['review_id']),
                 'example_id': str(i),
                 'excel_id':example['excel_id'],
                 'question': gen_question(general_aspect_label,aspects_questions,example,domain_name),
                 'category': example['category'],
                 'aspect': general_aspect_label,
                 'label': str(example['sentiment'])
                }
        
        dataset_ABSA[dataset_name].append(entry)

dataset_ABSA['train'][0:3]

# #### Adding ID labeles
for dataset_name, dataset in dataset_ABSA.items():
    for i, example in enumerate(dataset):
        example['guid'] = f'{domain_name}-{dataset_name}-r{example["review_id"]}-e{example["example_id"]}'


# #### Saving Data
for dataset_name, dataset in dataset_ABSA.items():
    
    with open(os.path.join(args.output_dir,f'{domain_name}_{dataset_name}.jsonl'), 'w') as f:
        for example in dataset:
            f.write(f'{example}\n')

print(f"Dataset is generated for the domain: {domain_name}")