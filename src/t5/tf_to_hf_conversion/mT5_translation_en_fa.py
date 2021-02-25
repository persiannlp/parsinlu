from transformers import T5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if True:
    model_name = "google/mt5-xl"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(T5Config.from_pretrained(model_name))

    load_tf_weights_in_t5(model, None, "/Users/danielk/ideaProjects/parsiglue-baselines/huggingface_example_scripts/mt5/xl")
    model.eval()

    model.save_pretrained(f"/Users/danielk/ideaProjects/parsiglue-baselines/huggingface_example_scripts/mT5_persian_multiple_choice_xl")
    tokenizer.save_pretrained(f"/Users/danielk/ideaProjects/parsiglue-baselines/huggingface_example_scripts/mT5_persian_multiple_choice_xl")
else:
    model_name = "/Users/danielk/ideaProjects/parsiglue-baselines/huggingface_example_scripts/mT5_persian_multiple_choice_small"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("وسیع ترین کشور جهان کدام است؟ <sep> آمریکا <sep> کانادا <sep> روسیه <sep> چین")
run_model("طامع یعنی ؟ <sep> آزمند <sep> خوش شانس <sep> محتاج <sep> مطمئن")
run_model("زمینی به ۳۱ قطعه متساوی مفروض شده است و هر روز مساحت آماده شده برای احداث، دو برابر مساحت روز قبل است.اگر پس از (۵ روز) تمام زمین آماده شده باشد، در چه روزی یک قطعه زمین آماده شده <sep> روز اول <sep> روز دوم <sep> روز سوم <sep> هیچکدام")

