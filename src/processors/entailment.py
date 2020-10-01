import os

from transformers import InputExample, DataProcessor


class TEProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str), f"Training input {text_a} is not a string"
            assert isinstance(text_b, str), f"Training input {text_b} is not a string"
            assert isinstance(label, str), f"Training label {label} is not a string"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

