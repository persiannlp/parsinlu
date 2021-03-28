from transformers import MT5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if False:
    model_name = "google/mt5-large"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(MT5Config.from_pretrained(model_name))

    load_tf_weights_in_t5(model, None, "/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/large")
    model.eval()

    model.save_pretrained(f"/Users/danielk/ideaProjects/mt5-large-parsinlu-qqp-query-paraphrasing")
    tokenizer.save_pretrained(f"/Users/danielk/ideaProjects/mt5-large-parsinlu-qqp-query-paraphrasing")
else:
    model_name = "persiannlp/mt5-small-parsinlu-qqp-query-paraphrasing"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

def run_model(q1, q2, **generator_args):
    input_ids = tokenizer.encode(f"{q1}<sep>{q2}", return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("چه چیزی باعث پوکی استخوان می شود؟", "چه چیزی باعث مقاومت استخوان در برابر ضربه می شود؟")
run_model("من دارم به این فکر میکنم چرا ساعت هفت نمیشه؟", "چرا من ساده فکر میکردم به عشقت پابندی؟")
run_model("دعای کمیل در چه روزهایی خوانده می شود؟", "دعای جوشن کبیر در چه شبی خوانده می شود؟")
run_model("دعای کمیل در چه روزهایی خوانده می شود؟", "دعای جوشن کبیر در چه شبی خوانده می شود؟")
run_model("شناسنامه در چه سالی وارد ایران شد؟", "سیب زمینی در چه سالی وارد ایران شد؟")
run_model("سیب زمینی چه زمانی وارد ایران شد؟", "سیب زمینی در چه سالی وارد ایران شد؟")
