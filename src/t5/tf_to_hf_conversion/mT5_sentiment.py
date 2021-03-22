from transformers import T5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if False:
    size = "small"
    model_name = f"google/mt5-{size}"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(T5Config.from_pretrained(model_name))

    load_tf_weights_in_t5(model, None, f"/Users/danielk/ideaProjects/parsiglue-baselines/src/t5/tf_to_hf_conversion/{size}")
    model.eval()

    model.save_pretrained(f"/Users/danielk/ideaProjects/mt5-{size}-parsinlu-sentiment-analysis")
    tokenizer.save_pretrained(f"/Users/danielk/ideaProjects/mt5-{size}-parsinlu-sentiment-analysis")
else:
    model_name = "persiannlp/mt5-small-parsinlu-sentiment-analysis"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

def run_model(context, query, **generator_args):
    input_ids = tokenizer.encode(context + "<sep>" + query, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model(
    "یک فیلم ضعیف بی محتوا بدون فیلمنامه . شوخی های سخیف .",
    "نظر شما در مورد داستان، فیلمنامه، دیالوگ ها و موضوع فیلم  لونه زنبور چیست؟"
)

run_model(
    "فیلم تا وسط فیلم یعنی دقیقا تا جایی که معلوم میشه بچه های املشی دنبال رضان خیلی خوب و جذاب پیش میره ولی دقیقا از همونجاش سکته میزنه و خلاص...",
    "نظر شما به صورت کلی در مورد فیلم  ژن خوک چیست؟"
)
run_model(
    "اصلا به هیچ عنوان علاقه نداشتم اجرای می سی سی پی نشسته میمیرد روی پرده سینما ببینم  دیالوگ های تکراری   هلیکوپتر  ماشین  آلندلون  لئون  پاپیون  آخه چرااااااااااااااا   همون حسی که توی تالار وحدت بعد از نیم ساعت به سرم اومد امشب توی سالن سینما تجربه کردم ،حس گریز از سالن.......⁦ ⁦(ノಠ益ಠ)ノ⁩ ",
    " نظر شما در مورد صداگذاری و جلوه های صوتی فیلم  مسخره‌باز چیست؟"
)

run_model(
    " گول نخورید این رنگارنگ مینو نیست برای شرکت گرجیه و متاسفانه این محصولش اصلا مزه رنگارنگی که انتظار دارید رو نمیده ",
    " نظر شما در مورد عطر، بو، و طعم این بیسکویت و ویفر چیست؟"
)

run_model(
    "در مقایسه با سایر برندهای موجود در بازار با توجه به حراجی که داشت ارزانتر ب",
    " شما در مورد قیمت و ارزش خرید این حبوبات و سویا چیست؟"
)

run_model(
    "من پسرم عاشق ایناس ولی دیگه به خاطر حفظ محیط زیست فقط زمانهایی که مجبور باشم شیر دونه ای میخرم و سعی میکنم دیگه کمتر شیر با بسته بندی تتراپک استفاده کنم ",
    "نظر شما به صورت کلی در مورد این شیر چیست؟"
)
