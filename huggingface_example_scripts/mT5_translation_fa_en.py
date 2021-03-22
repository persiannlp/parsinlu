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
    model_size="small"
    model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("ستایش خدای را که پروردگار جهانیان است.")
run_model("در هاید پارک کرنر بر گلدانی ایستاده موعظه می‌کند؛")
run_model("وی از تمامی بلاگرها، سازمان‌ها و افرادی که از وی پشتیبانی کرده‌اند، تشکر کرد.")
run_model("مشابه سال ۲۰۰۱، تولید آمونیاک بی آب در ایالات متحده در سال ۲۰۰۰ تقریباً ۱۷،۴۰۰،۰۰۰ تن (معادل بدون آب) با مصرف ظاهری ۲۲،۰۰۰،۰۰۰ تن و حدود ۴۶۰۰۰۰۰ با واردات خالص مواجه شد. ")
run_model("می خواهم دکترای علوم کامپیوتر راجع به شبکه های اجتماعی را دنبال کنم، چالش حل نشده در شبکه های اجتماعی چیست؟")
