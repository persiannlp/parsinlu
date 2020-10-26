from processors.qqp import QQPProcessor
from processors.entailment import TEProcessor
from processors.ABSAProcessor import ABSAProcessor
from transformers.data.metrics import simple_accuracy

import logging

logger = logging.getLogger(__name__)

processors = {
    "qqp": QQPProcessor,
    "entailment": TEProcessor,
    "sentiment": ABSAProcessor
}

output_modes = {
    "qqp": "classification",
    "entailment": "classification",
    "sentiment": "classification"
}

tasks_num_labels = {
    "qqp": 2,
    "entailment": 3,
    "multiple_choice": 4,
    "sentiment": 7
}


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "qqp":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "entailment":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sentiment":
        print("******** EVAlUATING SENTIMENT ANALYSIS ***********\n")
        print(f"tpye of preds: {type(preds)}, type of labels: {type(labels)}\n\n")
        print(f"value of preds: {preds},\n\n value of labels: {labels}\n\n")
        return {"acc": 0.5}
    else:
        raise KeyError(task_name)


