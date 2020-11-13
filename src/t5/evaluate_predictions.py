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
import re

import numpy as np
import sacrebleu
import random
import tensorflow.compat.v1 as tf

from docopt import docopt
from google.cloud import storage
from tqdm import tqdm


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")

# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
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
            if eval_metric == 'multiple-choice':
                decoded_question = input[2:-1]  # drop the begining and ending characters
                # convert byte-encoded question to utf-8-encoded question
                decoded_question = decoded_question.encode('raw_unicode_escape'). \
                    decode('unicode_escape').encode("raw_unicode_escape").decode('utf-8')
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
            predictions = dict(zip(inputs, predictions))
            num_predictions = len(predictions)
            # if num_predictions != num_targets:
            #     print(f'Something is wrong! The no. of predictions does not match no. of target labels. num_targets: {num_targets} - num_predictions: {num_predictions}')
            score, num_examples, missing_examples = evaluate(targets, predictions, eval_metric, outfile, instance_subset_map)
            print(f'Score on current checkpoint: {score}')
            if score > best_score:
                best_score = score
                best_checkpoint = prediction_checkpoint.name
                best_checkpoint_predictions = predictions
                best_checkpoint_num_examples = num_examples
                best_checkpoint_missing_examples = missing_examples

        print(
            f'Evaluated all checkpoints. Best checkpoint: {best_checkpoint}. Best score: {best_score} from {best_checkpoint_num_examples} questions. No. of missing questions: {len(best_checkpoint_missing_examples)}')
