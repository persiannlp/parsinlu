from transformers import T5ForConditionalGeneration, AutoTokenizer

model_size = "small"
model_name = "persiannlp/mT5_persian_multiple_choice_" + model_size
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("وسیع ترین کشور جهان کدام است؟ <sep> آمریکا <sep> کانادا <sep> روسیه <sep> چین")
run_model("طامع یعنی ؟ <sep> آزمند <sep> خوش شانس <sep> محتاج <sep> مطمئن")
run_model("زمینی به ۳۱ قطعه متساوی مفروض شده است و هر روز مساحت آماده شده برای احداث، دو برابر مساحت روز قبل است.اگر پس از (۵ روز) تمام زمین آماده شده باشد، در چه روزی یک قطعه زمین آماده شده <sep> روز اول <sep> روز دوم <sep> روز سوم <sep> هیچکدام")

