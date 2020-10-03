import csv
from transformers import DataProcessor
import torch
from torch.utils.data.dataset import Dataset
import logging
import os
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


def read_csv(input_file):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


class MultipleChoiceProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def read_examples(self, name, data_dir):
        lines = read_csv(os.path.join(data_dir, name))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            id = "%s-%s-%s" % (line[0], str(i), name)

            # print(list(enumerate(line)))
            question = line[1]

            label = str(int(line[2]) - 1)
            if len(question) < 5:
                continue
            options = []
            for i in range(3, 7):
                if len(line) > i and len((line[i]).strip()) > 0:
                    options.append(line[i].strip())

            if len(options) < 2:
                continue

            for i in range(0, 4):
                if len(options) < 4:
                    options.append("")
                else:
                    break

            assert len(options) == 4

            assert isinstance(label, str), f"Training label {label} is not a string"
            example = InputExample(
                example_id=id,
                question=question,
                contexts=[
                    "",
                    "",
                    "",
                    "",
                ],
                endings=options,
                label=label,
            )
            if i < 50:
                print("----")
                print(f"question: {question}")
                print(f"label: {label}")
                print(f"options: {options}")
                print(example)
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self.read_examples("train.csv", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self.read_examples("dev.csv", data_dir)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self.read_examples("test.csv", data_dir)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = MultipleChoiceProcessor()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    # examples = processor.get_dev_examples(data_dir)
                    # TODO update this
                    examples = processor.get_train_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    label_list,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]

        # print(choices_inputs)
        # print(example)

        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features
#
# processors = {
#     # "qqp": QQPProcessor,
#     # "entailment": TEProcessor,
#     "multiple_choice": MultipleChoiceProcessor,
# }
