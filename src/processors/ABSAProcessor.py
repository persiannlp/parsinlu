#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import sys
sys.path.append('../')

import os
import json
from transformers import InputExample, DataProcessor
from sklearn.metrics import f1_score, accuracy_score


class ABSAProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        self.labels = ['-3','-2','-1','0','1','2','3']
        self.test_file_name = 'food_test.jsonl'
        self.dev_file_name = 'food_dev.jsonl'

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
        dataset = self.load_data_jsonl(os.path.join(data_dir,"ABSA_Dataset_train.jsonl"))
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

        dataset = self.load_data_jsonl(os.path.join(data_dir,self.dev_file_name))

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

            examples.append(example)

        return examples

    def get_test_examples(self, data_dir):
        """See base class."""

        dataset = self.load_data_jsonl(os.path.join(data_dir,self.test_file_name))

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


def separate_sentiment_aspect(data, first_id, num_aspects):
    num_review = len(data) // num_aspects
    overall_y, y = [], []
    for i in range(int(first_id), int(first_id)+num_review-1):
        aspects = []
        for j in range(num_aspects):
            idx = 'test-r{}-e{}'.format(i+1, j+1)
            if j+1 == num_aspects:
                overall_y.append(data[idx][1])
            else:
                aspects.append(data[idx])
        y.append(aspects)

    assert len(overall_y) == len(y), "error in separating overall sentiment from aspects"

    return overall_y, y

def eval_sentiment(y_true, y_pred):
    """
    Prediction accuracy (percentage) and F1 score for sentiment analysis task
    y_true: a list of pairs (aspect, polarity)
    y_pred: a list of pairs (aspect, polarity)
    """

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def eval_aspect_macro_f1(y_true, y_pred):
    """
    Calculate "Macro-F1" for aspect detection task
    y_true: actual labels
    y_pred: predictions
    """
    p_all = 0
    r_all = 0
    count = 0
    for i in range(len(y_pred)):
        a = set()
        b = set()
        for j in range(len(y_pred[i])):
            if y_pred[i][j][1] != 0:
                a.add(j)
            if y_true[i][j][1] != 0:
                b.add(j)
        if len(b) == 0: continue
        a_b = a.intersection(b)
        if len(a_b) > 0:
            p = len(a_b) / len(a)
            r = len(a_b) / len(b)
        else:
            p = 0
            r = 0
        count += 1
        p_all += p
        r_all += r
    Ma_p = p_all / count
    Ma_r = r_all / count
    aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)

    return aspect_Macro_F1


def eval_aspect_polarity_accuracy(y_true, y_pred, num_aspects):

    num_aspects = num_aspects-1
    total_cases = len(y_true)
    flag = False
    true_cases = 0
    for i in range(total_cases):
        for j in range(num_aspects):
            if y_true[i][j] != y_pred[i][j]: flag = True
        # if y_true[i][1] != y_pred[i][1]: continue
        # if y_true[i][2] != y_pred[i][2]: continue
        # if y_true[i][3] != y_pred[i][3]: continue
        # if y_true[i][4] != y_pred[i][4]: continue
        # if y_true[i][5] != y_pred[i][5]: continue
        if not flag:
            true_cases += 1
        flag = False
    aspect_strict_Acc = true_cases / total_cases

    return aspect_strict_Acc

def absa_evaluation(data_dir, output_ids, preds):

    label_map = {'-3': 0, '-2': 1, '-1': 2, '0': 3, '1': 4, '2': 5, '3': 6}

    y_true_samples = {}
    y_pred_samples = {}

    available_aspects = set()
    with open(os.path.join(data_dir,self.test_file_name), 'r') as file:
        lines = file.readlines()

    dataset = []
    for line in lines:
        data = json.loads(line.replace('\'', '\"'))
        dataset.append(data)
        available_aspects.add(data['example_id'])

    num_aspects = len(available_aspects)
    print("Number of aspects: {}".format(num_aspects))

    first_id = dataset[0]['review_id']
    for i, entry in enumerate(dataset):
        guid = 'test-r{}-e{}'.format(entry["review_id"],entry["example_id"])
        aspect = str(entry["aspect"])
        label = str(entry["label"])
        y_true_samples[guid] = [aspect, label_map[label]]
        pred_label = preds[(i//num_aspects) * num_aspects + int(entry["example_id"]) - 1]
        y_pred_samples[guid] = [aspect, pred_label]

    # for i in range(len(dataset)):
    #     for j in range(1,8):
    #         guid = 'test-r{}-e{}'.format(i+1, j)
    #         aspect = y_true_samples[guid][0]
    #         label = preds[i*7+j]
    #         y_pred_samples[guid] = [aspect, label]

    assert len(y_true_samples) == len(y_pred_samples), "pred size doesn't match with actual size"

    # Overall sentiment scores should be separated from the aspects
    overall_y_true, y_true = separate_sentiment_aspect(y_true_samples, first_id, num_aspects)
    overall_y_pred, y_pred = separate_sentiment_aspect(y_pred_samples, first_id, num_aspects)

    print(y_true, y_pred)
    print(overall_y_true, overall_y_pred)

    sentiment_acc, sentiment_macro_f1 = eval_sentiment(overall_y_true, overall_y_pred)
    aspect_macro_f1 = eval_aspect_macro_f1(y_true, y_pred)
    aspect_strict_acc = eval_aspect_polarity_accuracy(y_true, y_pred, num_aspects)

    return sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_strict_acc


# if __name__ == "__main__":
#     y_true = [[['طعم', -3], ['ارزش خرید', 2], ['کیفیت', 0]], [['طعم', 2], ['ارزش خرید', 1], ['کیفیت', 2]],
#               [['طعم', -3], ['ارزش خرید', -3], ['کیفیت', 0]]]
#
#     y_pred = [[['طعم', -3], ['ارزش خرید', 1], ['کیفیت', -3]], [['طعم', 2], ['ارزش خرید', 1], ['کیفیت', 2]],
#               [['طعم', 2], ['ارزش خرید', 1], ['کیفیت', -3]]]
#
#
#     print(eval_aspect_polarity_accuracy(y_true, y_pred, 3))
#     print(eval_aspect_macro_f1(y_true, y_pred))
#
    # y_true = [-2, 1, 0, 1, 2, 1, 1]
    # y_pred = [-2, 1, 0, 3, 2, 1, 1]
    #
    # print(eval_sentiment(y_true, y_pred))

    # y_true_samples = {}
    # y_pred_samples = {}
    #
    # with open(os.path.join("/Users/niloofar/PycharmProjects/parsiglue/data/sentiment-analysis", "food_test.jsonl"), 'r') as file:
    #     lines = file.readlines()
    # dataset = []
    # for line in lines:
    #     data = json.loads(line.replace('\'', '\"'))
    #     dataset.append(data)
    #
    # for i, entry in enumerate(dataset):
    #     guid = 'test-r{}-e{}'.format(entry["review_id"], entry["example_id"])
    #     # aspect = str(entry["acpect"])
    #     aspect = "طعم"
    #     label = str(entry["label"])
    #     y_true_samples[guid] = [aspect, label]
    #
    # print(y_true_samples)






