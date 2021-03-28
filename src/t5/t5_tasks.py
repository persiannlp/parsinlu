import t5
import os
import functools
import tensorflow as tf
from t5.data import sentencepiece_vocabulary
from t5.evaluation import metrics

DATA_DIR = "gs://danielk-files/data/"

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

DEFAULT_VOCAB = sentencepiece_vocabulary.SentencePieceVocabulary(
    DEFAULT_SPM_PATH)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}


def get_downloaded_data_path(data_dir1, split, extension):
    return os.path.join(data_dir1, split + extension)


def preprocess(
        dataset,
        prefix='',  # not used
        sample_answer=False,  # not used
):
    def data_map(ex):
        """Map Natural Questions example to text-to-text example."""
        input = ex['input']
        target = ex['target']

        return {'inputs': input, 'targets': target, 'answers': target}

    dataset = dataset.map(
        data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.filter(lambda ex: tf.strings.length(ex['targets']) > 0)


def dataset_fn(split, shuffle_files=False, dataset=""):
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(get_downloaded_data_path(DATA_DIR + dataset, split, ".tsv"))
    print(" >>>> about to read tsv . . . ")
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""], use_quote_delim=False, field_delim="\t"),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds


def postprocessor(answer, example=None, is_target=False):
    """Returns answer, or all answers if the full example is provided."""
    if is_target:
        return example["answers"]
    return answer


t5.data.TaskRegistry.add(
    f"qqp",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="qqp"),
    splits=["train", "dev", "test"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad]
)

t5.data.TaskRegistry.add(
    f"qqp_english",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="qqp_english"),
    splits=["qqp_english_train", "qqp_english_validation", "qqp_english_test"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad]
)

t5.data.TaskRegistry.add(
    f"multiple_choice_str",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="multiple_choice_str"),
    splits=["train", "valid", "test_ml", 'test_lit', 'test_ck'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"english_multiple_choice_arc_comqa_obqa",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="english_multiple_choice_arc_comqa_obqa"),
    splits=["train", "test", 'dev'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"translation_combined_fa_en",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="translation_combined_fa_en"),
    splits=["train", "test", 'dev'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"translation_combined_en_fa",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="translation_combined_en_fa"),
    splits=["train", "test", 'dev'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"arabic_english_opus100",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="arabic_english_opus100"),
    splits=["train", "test", 'dev'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"parsiglue_entailment",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="parsiglue_entailment"),
    splits=["train", "test_farstail", 'test_natural', 'test_translation', 'dev'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"snli_entailment",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="snli_entailment"),
    splits=["train", "dev", 'test'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"squad1_1",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="squad1_1"),
    splits=["train", "dev", 'test'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"parsiglue_sentiment",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="parsiglue_sentiment"),
    splits=["train", "merged_dev", 'movie_test', 'food_test'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"restuarant_reviews_english_dataset",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="parsiglue_sentiment"),
    splits=["train", "merged_dev", 'movie_test', 'food_test'],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)

t5.data.TaskRegistry.add(
    f"parsiglue_readingcomprehension",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=functools.partial(dataset_fn, dataset="parsiglue_readingcomprehension"),
    splits=["train", "dev", "eval"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=preprocess,
    # Lowercase targets before computing metrics.
    postprocess_fn=postprocessor,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad],
)


t5.data.MixtureRegistry.add(f"entailment_mixture", ["parsiglue_entailment", "snli_entailment"], default_rate=1.0)
t5.data.MixtureRegistry.add(f"translation_x_en_mixture", ["translation_combined_fa_en", "arabic_english_opus100"], default_rate=1.0)
t5.data.MixtureRegistry.add(f"multiple_choice_mixture", ["multiple_choice_str", "english_multiple_choice_arc_comqa_obqa"], default_rate=1.0)
t5.data.MixtureRegistry.add(f"qqp_mixture", ["qqp", "qqp_english"], default_rate=1.0)
t5.data.MixtureRegistry.add(f"reading_com_mixture", ["parsiglue_readingcomprehension", "squad1_1"], default_rate=1.0)
t5.data.MixtureRegistry.add(f"sentiment_mixture", ["parsiglue_sentiment", "restuarant_reviews_english_dataset"], default_rate=1.0)
