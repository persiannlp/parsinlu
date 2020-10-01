import os

from transformers import InputExample, DataProcessor


class QQPProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s-%s-%s" % (line[0], line[1], line[2], "train")
            text_a = line[6]
            text_b = line[7]
            if len(text_a) < 5 or len(text_b) < 5:
                continue
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
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s-%s-%s" % (line[0], line[1], line[2], "test")
            text_a = line[6]
            text_b = line[7]
            if len(text_a) < 5 or len(text_b) < 5:
                continue
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


