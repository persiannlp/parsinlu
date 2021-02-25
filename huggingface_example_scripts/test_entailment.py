from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

labels = ["entails", "contradicts", "neutral"]
model_name_or_path = "bert-base-multilingual-cased_batch_size=8_learning_rate=3e-5_learning_rate=3e-5_num_train_epoch=3"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,)


def model_predict(text_a, text_b):
    features = tokenizer( [(text_a, text_b)], padding="max_length", truncation=True, return_tensors='pt')
    output = model(**features)
    logits = output[0]
    print(logits)


model_predict(
    "این مسابقات بین آوریل و دسامبر در هیپودروم ولیفندی در نزدیکی باکرکی ، ۱۵ کیلومتری (۹ مایل) غرب استانبول برگزار می شود.",
    "در ولیفندی هیپودروم، مسابقاتی از آوریل تا دسامبر وجود دارد."
)

model_predict(
"آیا کودکانی وجود دارند که نیاز به سرگرمی دارند؟",
    "هیچ کودکی هرگز نمی خواهد سرگرم شود.",
)

model_predict(
    "ما به سفرهایی رفته ایم که در نهرهایی شنا کرده ایم",
    "علاوه بر استحمام در نهرها ، ما به اسپا ها و سونا ها نیز رفته ایم."
)

# model.to(args.device)
# model.save_pretrained("path/to/repo/clone/your-model-name")
# tokenizer.save_pretrained("path/to/repo/clone/your-model-name")