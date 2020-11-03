#!/usr/bin/env python

"""Evaluate model predictions against target.
Usage:
   evaluate_predictions.py --bucket_name=GOOGLE_CLOUD_BUCKET_NAME --eval_path=NAME --eval_metric=METRIC_NAME
   evaluate_predictions.py -h| --help
Options:
    -h --help                               Show this screen
   --bucket_name=GOOGLE_CLOUD_BUCKET_NAME   Name of the bucket in GoogleCloud where the predictions are stored
   --eval_path=NAME                         gs:// link to predictions to evaluate
   --eval_metric=METRIC_NAME                Name of the evaluation metric. Currently recognized options: efficientqa_exact, efficientqa_regex
"""
# import efficientqa_eval_utils
import re
import t5
import tensorflow.compat.v1 as tf

from docopt import docopt
from google.cloud import storage


def evaluate(targets, predictions, eval_metric):
    score = 0
    num_examples = 0
    missing_examples = []
    for question in targets:
        if question in predictions:
            curr_score = 0
            num_examples += 1
            if eval_metric == 'accuracy':
                curr_score = 0
                assert len(targets[question]) == 1
                if targets[question][0] == predictions[question]:
                    curr_score = 1.0
            score += curr_score
        else:
            missing_examples.append(question)
    avg_score = score / float(num_examples)
    return (avg_score, num_examples, missing_examples)


def get_lines_from_file(bucket_name, file_name):
    full_file_name = f'gs://{bucket_name}/{file_name}'
    lines = []
    with tf.io.gfile.GFile(full_file_name) as ip_lines:
        for line in ip_lines:
            lines.append(line.strip())
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

    # print(f'Bucket name: {bucket_name}, Path: {path}')
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(
        bucket_name, prefix=eval_path
    ))

    inputs_file = ([blob for blob in blobs if blob.name.endswith('_inputs')])[0]
    inputs = get_lines_from_file(bucket_name, inputs_file.name)
    targets_file = ([blob for blob in blobs if blob.name.endswith('_targets')])[0]
    targets = get_lines_from_file(bucket_name, targets_file.name)
    targets = [preprocess_target_line(target_line) for target_line in targets]
    targets = dict(zip(inputs, targets))

    eval_metric = args["--eval_metric"]

    num_targets = len(targets)
    if num_targets > 0:
        prediction_checkpoints = [blob for blob in blobs if blob.name.endswith('_predictions')]
        best_score = 0.0
        best_checkpoint = None
        best_checkpoint_num_examples = 0
        best_checkpoint_missing_examples = []
        best_checkpoint_predictions = []
        for prediction_checkpoint in prediction_checkpoints:
            print(f'Evaluating prediction checkpoint {prediction_checkpoint.name}')
            predictions = get_lines_from_file(bucket_name, prediction_checkpoint.name)
            predictions = dict(zip(inputs, predictions))
            num_predictions = len(predictions)
            if num_predictions != num_targets:
                print('Something is wrong! The no. of predictions does not match no. of target labels.')
            score, num_examples, missing_examples = evaluate(targets, predictions, eval_metric)
            print(f'Score on current checkpoint: {score}')
            if score > best_score:
                best_score = score
                best_checkpoint = prediction_checkpoint.name
                best_checkpoint_predictions = predictions
                best_checkpoint_num_examples = num_examples
                best_checkpoint_missing_examples = missing_examples
        print(
            f'Evaluated all checkpoints. Best checkpoint: {best_checkpoint}. Best score: {best_score} from {best_checkpoint_num_examples} questions. No. of missing questions: {len(best_checkpoint_missing_examples)}')
