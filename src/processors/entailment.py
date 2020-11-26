import os

from transformers import InputExample, DataProcessor


class TEProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        self.labels = ["n", "c", "n"]

    def read_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            input = line[0]
            guid = "%s-%s" % (input, "train")
            text_a = input.split("<sep>")[0]
            text_b = input.split("<sep>")[1]
            if len(text_a) < 5 or len(text_b) < 5:
                continue
            label = line[1]
            if label not in self.labels:
                continue

            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            if i < 10:
                print(example)
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        return self.read_examples(data_dir, "train.tsv")

    def get_test_examples(self, data_dir):
        return self.read_examples(data_dir, "test.tsv")

    def get_dev_examples(self, data_dir):
        return self.read_examples(data_dir, "dev.tsv")

    def get_labels(self):
        """See base class."""
        return self.labels


