from processors.qqp import QQPProcessor
from processors.entailment import TEProcessor
from processors.ABSAProcessor import ABSAProcessor, absa_evaluation
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


def compute_metrics(task_name, preds, labels, sample_ids=None, data_dir=None):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "qqp":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "entailment":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sentiment":
        print("******** EVAlUATING SENTIMENT ANALYSIS ***********\n")
        print(f"type of preds: {type(preds)}, type of labels: {type(labels)}\n\n")
        print(f"value of preds: {preds},\n\n value of labels: {labels}\n\n")
        sentiment_acc, sentiment_macro_f1, aspect_macro_f1, aspect_strict_acc = absa_evaluation(data_dir, sample_ids, preds)
        return {"SA_acc": sentiment_acc, "SA_macro_f1": sentiment_macro_f1, "absa_macro_f1": aspect_macro_f1,
                "absa_strict_acc": aspect_strict_acc}
    else:
        raise KeyError(task_name)


