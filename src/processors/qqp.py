import json
import os

from transformers import InputExample, DataProcessor


class QQPProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = []
        with open(os.path.join(data_dir, "train.jsonl"), "r") as f:
            for i, line in enumerate(f.readlines()):
                json_content = json.loads(line.replace("\n", ""))
                guid = "%s-%s" % (i, "train")
                text_a = json_content['q1']
                text_b = json_content['q2']
                assert label == '1' or label == '0'
                label = "paraphrase" if line[5] == "0" else "not-paraphrase"
                assert isinstance(text_a, str), f"Training input {text_a} is not a string"
                assert isinstance(text_b, str), f"Training input {text_b} is not a string"
                assert isinstance(label, str), f"Training label {label} is not a string"
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                if i < 10:
                    print(example)
                examples.append(example)
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = []
        with open(os.path.join(data_dir, "test.jsonl"), "r") as f:
            for i, line in enumerate(f.readlines()):
                json_content = json.loads(line.replace("\n", ""))
                guid = "%s-%s" % (i, "test")
                text_a = json_content['q1']
                text_b = json_content['q2']
                assert label == '1' or label == '0'
                label = "paraphrase" if line[5] == "0" else "not-paraphrase"
                assert isinstance(text_a, str), f"Training input {text_a} is not a string"
                assert isinstance(text_b, str), f"Training input {text_b} is not a string"
                assert isinstance(label, str), f"Training label {label} is not a string"
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                if i < 10:
                    print(example)
                examples.append(example)
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = []
        with open(os.path.join(data_dir, "deb.jsonl"), "r") as f:
            for i, line in enumerate(f.readlines()):
                json_content = json.loads(line.replace("\n", ""))
                guid = "%s-%s" % (i, "dev")
                text_a = json_content['q1']
                text_b = json_content['q2']
                assert label == '1' or label == '0'
                label = "paraphrase" if line[5] == "0" else "not-paraphrase"
                assert isinstance(text_a, str), f"Training input {text_a} is not a string"
                assert isinstance(text_b, str), f"Training input {text_b} is not a string"
                assert isinstance(label, str), f"Training label {label} is not a string"
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                if i < 10:
                    print(example)
                examples.append(example)
        return examples


    def get_labels(self):
        """See base class."""
        return ["paraphrase", "not-paraphrase"]


