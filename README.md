# ParsiGLUE
 - intro and what it is 
 - where to download the data 


## Baselines  
 - Download the data and include it in the `data/` directory.  
 - See the relevant section on how to train models for each task:   
    * [Textual entailment](#textual-entailment) 
    * [Query Paraphrasing](#query-paraphrasing) 
    * [Reading Comprehension](#reading-comprehension)
    * [Multiple-choice QA](#multiple-choice-qa)
    * [Machine Translation](#machine-translation) 
    * [Sentiment Analaysis](#sentiment-analysis) 
   
If you're using a GPU, make sure that you set the appropriate environmental variable. For example: 
```bash
export CUDA_VISIBLE_DEVICES=0
```    
   
### Textual Entailment 
Textual Entailment is the task of deciding whether a  whether two given questions are paraphrases of each other or not. 

Here are several examples: 

|   | Premise | Hypothesis |
| --- | :---: | :---: |
|  entailment | <p dir='rtl' align='right'> این مسابقات بین آوریل و دسامبر در هیپودروم ولیفندی در نزدیکی باکرکی ، ۱۵ کیلومتری (۹ مایل) غرب استانبول برگزار می شود. </p>  | <p dir='rtl' align='right'> در ولیفندی هیپودروم، مسابقاتی از آوریل تا دسامبر وجود دارد. </p> |
|  contradiction | <p dir='rtl' align='right'> آیا کودکانی وجود دارند که نیاز به سرگرمی دارند؟ </p> | <p dir='rtl' align='right'> هیچ کودکی هرگز نمی خواهد سرگرم شود. </p> |
|  neutral | <p dir='rtl' align='right'> ما به سفرهایی رفته ایم که در نهرهایی شنا کرده ایم </p> | <p dir='rtl' align='right'> علاوه بر استحمام در نهرها ، ما به اسپا ها و سونا ها نیز رفته ایم. </p> |


 This example code fine-tunes mBERT (multi-lingual BERT) on the this task. 
 It runs in 10 mins on a single GeForce RTX 2080. 

```bash 
export DATA_DIR=data/entailment

python src/run_text_classification.py \
  --data_dir $DATA_DIR \
  --task_name entailment \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir ../models/ \
  --save_steps -1 \
  --overwrite_output_dir
```

Training with the previously defined hyper-parameters yields the following results on the test set:
 
```
acc = 0.3697560975609756
```

To reproduce our numbers with all our baselines, try [`train_and_evaluate_entailment_baselines.sh`](scripts/train_and_evaluate_entailment_baselines.sh) script.

 
 ### Query Paraphrasing 
 QQP is the task of detecting whether two given questions are paraphrases of each other or not.
 
 Here are several examples:  

|  Label | Question 1 | Question 2 |
| :---: | :---: | :---: |
|  not-paraphrase | <p dir='rtl' align='right'>さ あ ひ る به چه معنی است؟</p>  | <p dir='rtl' align='right'> &脑 洞 大 به چه معنی است؟</p> |
|  paraphrase | <p dir='rtl' align='right'> قانون سوم حرکت نیوتن چیست؟ آیا می توانید یک عمل و یک عکس العمل را با مثال توضیح دهید؟ </p>| <p dir='rtl' align='right'> آیا کسی می تواند قانون سوم حرکت نیوتون را توضیح دهد؟ </p> |
|  not-paraphrase | <p dir='rtl' align='right'> آیا لیزر موهای زائد باعث فرار دائمی از موهای ناخواسته می شود؟ </p>| <p dir='rtl' align='right'> آیا لیزر موهای زائد دائمی است؟ </p> |
|  paraphrase | <p dir='rtl' align='right'> چه شانس هایی وجود دارد که اگر هیلاری در انتخابات رأی عمومی به پیروزی برسد ، کالح انتخاباتی بر ضد ترامپ تصمیم بگیرد؟ </p>|<p dir='rtl' align='right'> این احتمال وجود دارد که در ۱۹ دسامبر ، کالج انتخاباتی بتواند دونالد ترامپ را از دور خارج کند و به هیلاری کلینتون رأی دهد؟ </p> |

 This example code fine-tunes mBERT (multi-lingual BERT) on the this task. 
 It should not take more than 10 mins on a single GeForce RTX 2080 GPU. 

```bash 
export DATA_DIR=../data/qqp

python src/run_text_classification.py \
  --data_dir $DATA_DIR \
  --task_name qqp \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir ../models/ \
  --save_steps -1
```

Training with the previously defined hyper-parameters yields the following results on the test set:
```
acc = 0.7237936772046589
```

To reproduce our numbers with all our baselines, try [`train_and_evaluate_qqp_baselines.sh`](scripts/train_and_evaluate_qqp_baselines.sh) script. 

 
### Reading Comprehension 
 TODO 
 
 ### Multiple-Choice QA 
 Here the task is to pick a correct answer among 3-5 given candidate answers.
 Here are several examples: 

|  Question | Correct Answer | Candidate 1 | Candidate 2 | Candidate 3 | Candidate 4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  پایتخت کشور استرالیا کدام است؟ | 3 | ملبورن | سیدنی | کنبرا |  |
|  منظومه یا مجموعه عناصر و اجزائی که با هم کنش و واکنش و ارتباط متقابل دارند را چه می نامند؟ | 4 | نهاد | سازمان اجتماعی | گشتالت | سیستم |
|  کدام یک از موارد زیر جزء مراحل چهارگانه تصمیم گیری، نمی باشد؟ | 3 | تعریف و تشخیص مشکل | دستیابی به راح حل ها | هدایت و نظارت | اجرای تصممیم |
|  مفهوم کلی کدام بیت با سایر ابیات متفاوت است؟ | 4 | از خلاف آمد عادت بطلب کام که من          کسب جمعیت از آن زلف پریشان کردم | گفتم که بوی زلفت گمراه عالمم کرد        گفتا اگر بدانی هم اوت رهبر آید | زلف آشفته‌ی او موجب جمعیت ماست      چون چنین است پس آشفته‌ترش باید کرد | اگر به زلف دراز تو دست ما نرسد       گناه بخت پریشان و دست کوته ماست |
|  یک مجسمه، یک گلدان و یک ساعت را که از لحاظ حجم تقریبا به یک اندازه می‌باشند به چند صورت مختلف می‌توان دو بدو در کنار هم و روی یک کمد برای زینت اتاق قرار داد؟ | 1 | ۶ | ۴ | ۲ | ۸ |

To train a model, here is a sample script.  
    
 
```bash 
export DATA_DIR=data/multiple-choice/
python src/run_multiple_choice.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --data_dir DATA_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --output_dir models/ \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps 2 \
    --overwrite_output
```

Training with the previously defined hyper-parameters yields the following results on the test set:
```
????
```

To reproduce our numbers with all our baselines, try [`train_and_evaluate_multiple_choice_baselines.sh`](scripts/train_and_evaluate_multiple_choice_baselines.sh) script.
 
 ### Machine Translation 
 TODO 
 
 ### Sentiment Analysis 
TODO 

## FAQ 
**I have GPU on my machine by `n_gpu` is shown as `0`. Where is the problem?** Check out [this thread](https://github.com/pytorch/pytorch/issues/15612).  

## Citation 
If you find this work useful please cite the following work: 
```bibtex 
@article{2020parsiglue,
    title={},
    author={},
    journal={arXiv},
    year={2020}
}
```