#!/usr/bin/env python

"""Evaluate model predictions against target.
Usage:
   evaluate_predictions.py --bucket_name=GOOGLE_CLOUD_BUCKET_NAME --eval_path=NAME --eval_metric=METRIC_NAME [--dump]
   evaluate_predictions.py -h| --help
Options:
    -h --help                               Show this screen
   --bucket_name=GOOGLE_CLOUD_BUCKET_NAME   Name of the bucket in GoogleCloud where the predictions are stored
   --eval_path=NAME                         gs:// link to predictions to evaluate
   --eval_metric=METRIC_NAME                Name of the evaluation metric. Currently recognized options: efficientqa_exact, efficientqa_regex
   --dump                                   If specified, we would write the prediction to disk (for multiple-choice questions)
"""
# import efficientqa_eval_utils
import numpy as np
import sacrebleu
import random
import tensorflow.compat.v1 as tf
from sklearn.metrics import f1_score, accuracy_score
from docopt import docopt
from google.cloud import storage
from tqdm import tqdm

import string
import re
import json
from collections import Counter


def separate_sentiment_aspect(data, first_id, num_aspects):
    num_review = len(data) // num_aspects
    overall_y, y = [], []
    for i in range(int(first_id), int(first_id) + num_review - 1):
        aspects = []
        for j in range(num_aspects):
            idx = 'test-r{}-e{}'.format(i + 1, j + 1)
            if j + 1 == num_aspects:
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
    num_aspects = num_aspects - 1
    total_cases = len(y_true)
    flag = False
    true_cases = 0
    for i in range(total_cases):
        for j in range(num_aspects):
            if y_true[i][j] != y_pred[i][j]: flag = True

        if not flag:
            true_cases += 1
        flag = False
    aspect_strict_Acc = true_cases / total_cases

    return aspect_strict_Acc


def absa_evaluation(test_file_name, preds):
    label_map2 = {
        'no sentiment expressed': '-3',
        'very negative': '-2',
        'negative': '-1',
        'neutral': '0',
        'positive': '1',
        'very positive': '2',
        'mixed':'3',
    }
    label_map = {'-3': 0, '-2': 1, '-1': 2, '0': 3, '1': 4, '2': 5, '3': 6}
    preds = [label_map[label_map2.get(p, str(random.randint(-3, 3)))] for p in preds]

    y_true_samples = {}
    y_pred_samples = {}

    available_aspects = set()
    with open(test_file_name, 'r') as file:
        lines = file.readlines()

    dataset = []
    for line in lines:
        data = json.loads(line.replace('\'', '\"'))
        dataset.append(data)
        available_aspects.add(data['example_id'])

    num_aspects = len(available_aspects)
    print("Number of aspects: {}: {}".format(num_aspects, available_aspects))

    first_id = dataset[0]['review_id']
    for i, entry in enumerate(dataset):
        guid = 'test-r{}-e{}'.format(entry["review_id"], entry["example_id"])
        aspect = str(entry["aspect"])
        label = str(entry["label"])
        y_true_samples[guid] = [aspect, label_map[label]]
        idx = (i // num_aspects) * num_aspects + int(entry["example_id"]) - 1
        pred_label = preds[idx]
        y_pred_samples[guid] = [aspect, pred_label]

    assert len(y_true_samples) == len(y_pred_samples), "pred size doesn't match with actual size"

    # Overall sentiment scores should be separated from the aspects
    overall_y_true, y_true = separate_sentiment_aspect(y_true_samples, first_id, num_aspects)
    overall_y_pred, y_pred = separate_sentiment_aspect(y_pred_samples, first_id, num_aspects)

    sentiment_acc, sentiment_macro_f1 = eval_sentiment(overall_y_true, overall_y_pred)
    aspect_macro_f1 = eval_aspect_macro_f1(y_true, y_pred)
    aspect_strict_acc = eval_aspect_polarity_accuracy(y_true, y_pred, num_aspects)

    return sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_strict_acc


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_squad(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


def convert_byte_code_to_utf(input):
    # convert byte-encoded question to utf-8-encoded question
    return input.encode('raw_unicode_escape'). \
        decode('unicode_escape').encode("raw_unicode_escape").decode('utf-8')


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0  # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def evaluate(targets, predictions, eval_metric, outfile, instance_subset_map):
    score = 0
    if instance_subset_map:
        score_per_subset = {}
    if eval_metric == 'sentiment':
        score_per_review = {}
    num_examples = 0
    missing_examples = []
    if outfile:
        outfile.write(f"decoded_question\ttarget\tprediction\tselected-candidate\n")
    if len(targets) > 10000:
        targets = random.sample(targets, 5000)
    for input, target in tqdm(targets):
        if input in predictions:
            curr_score = 0
            num_examples += 1
            if eval_metric == 'accuracy':
                curr_score = 0
                assert len(target) == 1
                if target[0] == predictions[input]:
                    curr_score = 1.0
            if eval_metric == 'reading_comprehension':
                if ('["' in target or "['" in target) and ']' in target:
                    target_json = json.loads(target)
                else:
                    target_json = [target]
                assert type(target_json) == list
                curr_score = metric_max_over_ground_truths(
                    f1_score_squad, predictions[input], target_json)

            if eval_metric == 'sentiment':
                print("--------")
                print(input)
                pred = predictions[input]
                input = input.split("<sep>")[0]
                if input not in score_per_review:
                    score_per_review[input] = []
                score_per_review[input].append(
                    [pred, target]
                )

            if eval_metric == 'multiple-choice':
                decoded_question = input[2:-1]  # drop the begining and ending characters
                # convert byte-encoded question to utf-8-encoded question
                decoded_question = convert_byte_code_to_utf(decoded_question)
                candidates = [x.strip() for x in decoded_question.split("<sep>")[1:]]
                scores = [score_string_similarity(x, predictions[input]) for x in candidates]
                selected_idx = np.argmax(scores) + 1
                if outfile:
                    outfile.write(f"{decoded_question}\t{target}\t{predictions[input]}\t{selected_idx}\n")
                curr_score = 0
                if replace_punctuation(target) == replace_punctuation(predictions[input]):
                    curr_score = 1.0
            if eval_metric == 'translation':
                if "///" in target:
                    target = [[x] for x in target.split("///")]
                else:
                    target = [[target]]
                pred = [predictions[input]]
                bleu = sacrebleu.corpus_bleu(pred, target, tokenize="intl", lowercase=True)
                if instance_subset_map:
                    subset = instance_subset_map[input]
                    if subset not in score_per_subset:
                        score_per_subset[subset] = []
                    score_per_subset[subset].append(bleu.score)

                if outfile:
                    outfile.write(f"{target}\t{predictions[input]}\t{bleu.score}\n")

                curr_score = bleu.score
            score += curr_score
        else:
            missing_examples.append(input)
    avg_score = score / float(num_examples)
    if instance_subset_map:
        for k, v in score_per_subset.items():
            avg_val = sum(v) / len(v)
            print(f" --subset--> {k}: {avg_val} ")

    return (avg_score, num_examples, missing_examples)


def get_lines_from_file(bucket_name, file_name):
    full_file_name = f'gs://{bucket_name}/{file_name}'
    lines = []
    with tf.io.gfile.GFile(full_file_name) as ip_lines:
        for line in ip_lines:
            lines.append(line.strip())
    print(f" * Loaded file `{file_name}` with length: {len(lines)}")
    return lines


# takes input string in the format:
# ['Dai Yongge', 'Yongge Dai', 'Xiu Li Dai', 'Dai Xiuli']
# and returns a list of strings from it
def preprocess_target_line(target_line):
    target_line = target_line.lstrip('[').rstrip(']')
    target_line = target_line.replace('", "', "', '").replace('\', "', "', '").replace('", \'', "', '")
    target_line_components = target_line.split("', '")
    targets = []
    for target_line_component in target_line_components:
        # drop the begining `b'` and the ending `'`
        target_line_component = str(target_line_component.strip("'").strip('"').strip())
        target_line_component = target_line_component[2:]
        targets.append(target_line_component)
    return targets


if __name__ == "__main__":
    inputs = []
    targets = {}

    # parse command line arguments
    args = docopt(__doc__)

    bucket_name = args["--bucket_name"]
    eval_path = args["--eval_path"]
    dump = args["--dump"]

    print(f'-------- \n Bucket name: {bucket_name}, Path: {eval_path} \n-------- ')
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(
        bucket_name, prefix=eval_path
    ))

    inputs_file = ([blob for blob in blobs if blob.name.endswith('_inputs')])[0]
    inputs = get_lines_from_file(bucket_name, inputs_file.name)
    targets_file = ([blob for blob in blobs if blob.name.endswith('_targets')])[0]
    targets = get_lines_from_file(bucket_name, targets_file.name)

    targets = list(zip(inputs, targets))

    eval_metric = args["--eval_metric"]

    if ("translation" in eval_path and 'test_eval' in eval_path) or "arabic_english_opus100" in eval_path:
        type = eval_path.split("/")[1]
        if "arabic_english_opus100" in eval_path:
            type = "translation_combined_fa_en"
        categories = []
        with open(f"/Users/danielk/ideaProjects/parsiglue-baselines/data/translation/{type}/test.tsv", "r") as f:
            instance_subsets = [line.split("\t")[-1].replace("\n", "") for line in f.readlines()]
            assert len(instance_subsets) == len(targets), f"{len(instance_subsets)} vs {len(targets)}: type: {type}"
            print(f" ------- \n * available types: {set(instance_subsets)} \n ------- ")
            instance_subset_map = dict(zip(inputs, instance_subsets))
    else:
        instance_subset_map = None

    outfile = None
    num_targets = len(targets)
    if num_targets > 0:
        prediction_checkpoints = [blob for blob in blobs if blob.name.endswith('_predictions')]
        # prediction_checkpoints.reverse()
        best_score = 0.0
        best_checkpoint = None
        best_checkpoint_num_examples = 0
        best_checkpoint_missing_examples = []
        best_checkpoint_predictions = []
        for prediction_checkpoint in prediction_checkpoints:
            print(f'Evaluating prediction checkpoint {prediction_checkpoint.name}')
            if dump:
                import os

                dir = "/".join(prediction_checkpoint.name.split("/")[:-1])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                outfile = open(prediction_checkpoint.name, "w")
            predictions = get_lines_from_file(bucket_name, prediction_checkpoint.name)
            if eval_metric != "sentiment":
                predictions = dict(zip(inputs, predictions))
                score, num_examples, missing_examples = evaluate(targets, predictions, eval_metric, outfile,
                                                                 instance_subset_map)
            else:
                try:
                    sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_sent_acc = absa_evaluation(
                        "../../data/sentiment-analysis/movie_test.jsonl",
                        predictions)
                    print(sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_sent_acc)
                    score = sentiment_acc
                except:
                    score = 0.0
                num_examples = len(predictions)
                missing_examples = 0


            print(f'Score on current checkpoint: {score}')
            if score > best_score:
                best_score = score
                best_checkpoint = prediction_checkpoint.name
                best_checkpoint_predictions = predictions
                best_checkpoint_num_examples = num_examples
                best_checkpoint_missing_examples = missing_examples

        print(
            f'Evaluated all checkpoints. Best checkpoint: {best_checkpoint}. Best score: {best_score} from {best_checkpoint_num_examples} questions. No. of missing questions: {len(best_checkpoint_missing_examples)}')
