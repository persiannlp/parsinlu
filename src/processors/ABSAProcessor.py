#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import sys
sys.path.append('../')

import os
import json
from transformers import InputExample, DataProcessor
from sklearn.metrics import f1_score, accuracy_score

test_file_name = 'movie_test.jsonl'
dev_file_name = 'movie_dev.jsonl'

class ABSAProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        self.labels = ['-3','-2','-1','0','1','2','3']
        self.test_file_name = test_file_name
        self.dev_file_name = dev_file_name

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


def sentiment_aspect_separator(data, first_id, num_aspects):
    """
    Separates overall sentiment scores from aspect sentiment scores
    first_id: smallest example id in the data
    num_aspects: number of aspects we have for a specific domain
    """

    num_review = len(data) // num_aspects
    SA_y, ABSA_y = [], []
    for i in range(int(first_id), int(first_id)+num_review-1):
        aspects = []
        for j in range(num_aspects):
            idx = 'test-r{}-e{}'.format(i+1, j+1)
            if j+1 == num_aspects:
                SA_y.append(data[idx][1])
            else:
                aspects.append(data[idx])
        ABSA_y.append(aspects)

    assert len(SA_y) == len(ABSA_y), "Error in separating overall sentiment from aspects"

    return SA_y, ABSA_y

def overall_sentiment_eval(y_true, y_pred):
    """
    Prediction accuracy (percentage) and F1 score for sentiment analysis task
    y_true: a list of pairs (aspect, polarity)
    y_pred: a list of pairs (aspect, polarity)
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, macro_f1

def aspect_extraction_eval(y_true, y_pred):
    """
    Calculate "Macro-F1" for aspect detection task
    y_true: actual labels
    y_pred: predictions
    """
    precision_all = 0
    recall_all = 0
    count = 0
    for i in range(len(y_pred)):
        predicted_aspects = set()
        actual_aspects = set()
        for j in range(len(y_pred[i])):
            if y_pred[i][j][1] != 0:
                predicted_aspects.add(j)
            if y_true[i][j][1] != 0:
                actual_aspects.add(j)
        if len(actual_aspects) == 0: continue
        common_aspects = predicted_aspects.intersection(actual_aspects)
        if len(common_aspects) > 0:
            precision = len(common_aspects) / len(predicted_aspects)
            recall = len(common_aspects) / len(actual_aspects)
        else:
            precision = 0
            recall = 0
        count += 1
        precision_all += precision
        recall_all += recall
    macro_precision = precision_all / count
    macro_recall = recall_all / count
    aspect_macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    return aspect_macro_f1


def aspect_polarity_accuracy_eval(y_true, y_pred, num_aspects):
    """
    For how many reviews both extracted aspects and their polarities are predicted correctly
    """

    num_aspects = num_aspects-1
    num_reviews = len(y_true)
    flag = False
    correct_preds = 0
    for i in range(num_reviews):
        for j in range(num_aspects):
            if y_true[i][j] != y_pred[i][j]: flag = True
        # if y_true[i][1] != y_pred[i][1]: continue
        # if y_true[i][2] != y_pred[i][2]: continue
        # if y_true[i][3] != y_pred[i][3]: continue
        # if y_true[i][4] != y_pred[i][4]: continue
        # if y_true[i][5] != y_pred[i][5]: continue
        if not flag:
            correct_preds += 1
        flag = False
    aspect_strict_accuracy = correct_preds / num_reviews

    return aspect_strict_accuracy

def absa_evaluation(data_dir, output_ids, preds, test_eval):

    label_map = {'-3': 0, '-2': 1, '-1': 2, '0': 3, '1': 4, '2': 5, '3': 6}

    y_true_samples = {}
    y_pred_samples = {}

    if test_eval:
        eval_file_name = test_file_name
    else:
        eval_file_name = dev_file_name

    available_aspects = set()
    with open(os.path.join(data_dir,eval_file_name), 'r') as file:
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
    actual_sentiment, actual_aspect_sentiment = sentiment_aspect_separator(y_true_samples, first_id, num_aspects)
    predicted_sentiment, predicted_aspect_sentiment = sentiment_aspect_separator(y_pred_samples, first_id, num_aspects)

    # print(actual_aspect_sentiment, predicted_aspect_sentiment)
    # print(actual_sentiment, predicted_sentiment)

    sentiment_acc, sentiment_macro_f1 = overall_sentiment_eval(actual_sentiment, predicted_sentiment)
    aspect_macro_f1 = aspect_extraction_eval(actual_aspect_sentiment, predicted_aspect_sentiment)
    aspect_strict_acc = aspect_polarity_accuracy_eval(actual_aspect_sentiment, predicted_aspect_sentiment, num_aspects)

    return sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_strict_acc