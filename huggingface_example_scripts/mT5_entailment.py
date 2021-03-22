from transformers import MT5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if False:
    size = "large"
    model_name = f"google/mt5-{size}"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(MT5Config.from_pretrained(model_name))

    load_tf_weights_in_t5(model, None, f"/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/{size}")
    model.eval()

    model.save_pretrained(f"/Users/danielk/ideaProjects/mt5-{size}-parsinlu-opus-translation_en_fa")
    tokenizer.save_pretrained(f"/Users/danielk/ideaProjects/mt5-{size}-parsinlu-opus-translation_en_fa")
else:
    model_size="base"
    model_name = f"persiannlp/mt5-{model_size}-parsinlu-snli-entailment"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)


def run_model(premise, hypothesis, **generator_args):
    input_ids = tokenizer.encode(f"{premise}<sep>{hypothesis}", return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model(
    "این مسابقات بین آوریل و دسامبر در هیپودروم ولیفندی در نزدیکی باکرکی ، ۱۵ کیلومتری (۹ مایل) غرب استانبول برگزار می شود.",
    "در ولیفندی هیپودروم، مسابقاتی از آوریل تا دسامبر وجود دارد."
)

run_model(
"آیا کودکانی وجود دارند که نیاز به سرگرمی دارند؟",
    "هیچ کودکی هرگز نمی خواهد سرگرم شود.",
)

run_model(
    "ما به سفرهایی رفته ایم که در نهرهایی شنا کرده ایم",
    "علاوه بر استحمام در نهرها ، ما به اسپا ها و سونا ها نیز رفته ایم."
)
