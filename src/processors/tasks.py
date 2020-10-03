from processors.qqp import QQPProcessor
from processors.entailment import TEProcessor
from transformers.data.metrics import simple_accuracy

import logging

logger = logging.getLogger(__name__)

processors = {
    "qqp": QQPProcessor,
    "entailment": TEProcessor,
}

output_modes = {
    "qqp": "classification",
    "entailment": "classification",
}

tasks_num_labels = {
    "qqp": 2,
    "entailment": 3,
    "multiple_choice": 4
}


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "qqp":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "entailment":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
