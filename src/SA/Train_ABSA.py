#!/usr/bin/env python
# coding: utf-8

# In[6]:


## Author: Arman Kabiri
## Date: Oct. 24, 2020


# In[48]:


import sys

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import datasets
import numpy as np
import tensorflow as tf

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    TFAutoModelForSequenceClassification,
    TFTrainer,
    TFTrainingArguments,
)


# ### Data Loader Class

# In[10]:


def get_tfds(
    train_file: str,
    eval_file: str,
    test_file: str,
    tokenizer: PreTrainedTokenizer,
    label_column_id: int,
    max_seq_length: Optional[int] = None,
):
    files = {}

    if train_file is not None:
        files[datasets.Split.TRAIN] = [train_file]
    if eval_file is not None:
        files[datasets.Split.VALIDATION] = [eval_file]
    if test_file is not None:
        files[datasets.Split.TEST] = [test_file]

    ds = datasets.load_dataset("csv", data_files=files)
    features_name = list(ds[list(files.keys())[0]].features.keys())
    label_name = features_name.pop(label_column_id)
    label_list = list(set(ds[list(files.keys())[0]][label_name]))
    label2id = {label: i for i, label in enumerate(label_list)}
    input_names = ["input_ids"] + tokenizer.model_input_names
    transformed_ds = {}

    if len(features_name) == 1:
        for k in files.keys():
            transformed_ds[k] = ds[k].map(
                lambda example: tokenizer.batch_encode_plus(
                    example[features_name[0]], truncation=True, max_length=max_seq_length, padding="max_length"
                ),
                batched=True,
            )
    elif len(features_name) == 2:
        for k in files.keys():
            transformed_ds[k] = ds[k].map(
                lambda example: tokenizer.batch_encode_plus(
                    (example[features_name[0]], example[features_name[1]]),
                    truncation=True,
                    max_length=max_seq_length,
                    padding="max_length",
                ),
                batched=True,
            )

    def gen_train():
        for ex in transformed_ds[datasets.Split.TRAIN]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    def gen_val():
        for ex in transformed_ds[datasets.Split.VALIDATION]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    def gen_test():
        for ex in transformed_ds[datasets.Split.TEST]:
            d = {k: v for k, v in ex.items() if k in input_names}
            label = label2id[ex[label_name]]
            yield (d, label)

    train_ds = (
        tf.data.Dataset.from_generator(
            gen_train,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.TRAIN in transformed_ds
        else None
    )

    if train_ds is not None:
        train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.TRAIN])))

    val_ds = (
        tf.data.Dataset.from_generator(
            gen_val,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.VALIDATION in transformed_ds
        else None
    )

    if val_ds is not None:
        val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.VALIDATION])))

    test_ds = (
        tf.data.Dataset.from_generator(
            gen_test,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )
        if datasets.Split.TEST in transformed_ds
        else None
    )

    if test_ds is not None:
        test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(len(ds[datasets.Split.TEST])))

    return train_ds, val_ds, test_ds, label2id


logger = logging.getLogger(__name__)


# ### Arguments Classes

# In[11]:


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    label_column_id: int = field(metadata={"help": "Which column contains the label"})
    train_file: str = field(default=None, metadata={"help": "The path of the training file"})
    dev_file: Optional[str] = field(default=None, metadata={"help": "The path of the development file"})
    test_file: Optional[str] = field(default=None, metadata={"help": "The path of the test file"})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


# In[12]:


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


# ### Main

# ##### Setting Arguments for notebook environment

# In[61]:


sys.argv = [
    '--Train_ABSA.py',
    '--train_file', 'train.csv', ### training dataset file location (mandatory if running with --do_train option)
    '--dev_file', 'dev.csv', ### development dataset file location (mandatory if running with --do_eval option)
    '--test_file', 'test.csv', ### test dataset file location (mandatory if running with --do_predict option)
    '--label_column_id', '0', ### which column corresponds to the labels
    '--model_name_or_path', 'bert-base-multilingual-uncased',
    '--output_dir', 'model',
    '--num_train_epochs', '4',
    '--per_device_train_batch_size', '16',
    '--per_device_eval_batch_size', '32',
    '--do_train',
    '--do_eval',
    '--do_predict',
    '--logging_steps', '10',
    '--evaluation_strategy', 'steps',
    '--save_steps', '10',
    '--overwrite_output_dir',
    '--max_seq_length', '128'
]


# In[62]:


# %tb
# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


# In[63]:


if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    print("adasdasd")
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

# print('Done')


# In[ ]:




