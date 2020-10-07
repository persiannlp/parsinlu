# ParsiGLUE
ParsiGLUE is a comprehensive suit of high-level NLP tasks for Persian language. 
This suit contains 6 different key NLP tasks --- *Reading Comprehension*, *Multiple-Choice Question-Answering*, 
*Textual Entailment*, *Sentiment Analysis*, *Query Paraphrasing* and *Machine Translation*. 

These challenges are collected by expert annotators native in Persian language and from various sources. 
If you'd like to see additional details on the dataset and how we collected, please check out [our publication](#citation).    
  
## Getting the data 
You can find the data under the [`data/`](data) directory.  

## Leaderboard
On our [leaderboard page](https://parsiglue.com) we host the state-of-art scores for each challenge task.  
 
## Baselines  
First, make sure you have the data your `data/` directory.

Set up your environment. You can use `conda` or `virtualenv` to create a Python environment. 
We have tested the code with Python3.7, but it should work on any version >= 3.5.
Make sure to install all the necessary dependencies: 
```bash 
> python install -r requirements.txt
```

If you're using a GPU, make sure that you set the appropriate environmental variable. For example: 
```bash
export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID # for example, YOUR_GPU_ID can be 0 for using your first GPU; or 0,1 if you're using two GPUs
```    
  
See the relevant section on how to train models for each task:   
    * [Textual entailment](#textual-entailment) 
    * [Query Paraphrasing](#query-paraphrasing) 
    * [Reading Comprehension](#reading-comprehension)
    * [Multiple-choice QA](#multiple-choice-qa)
    * [Machine Translation](#machine-translation) 
    * [Sentiment Analaysis](#sentiment-analysis) 
   
### Textual Entailment 
Textual Entailment is the task of deciding whether a  whether two given questions are paraphrases of each other or not. 

Here are several examples: 

|   | Premise | Hypothesis |
| --- | :---: | :---: |
|  entailment | <p dir='rtl' align='right'> این مسابقات بین آوریل و دسامبر در هیپودروم ولیفندی در نزدیکی باکرکی ، ۱۵ کیلومتری (۹ مایل) غرب استانبول برگزار می شود. </p>  | <p dir='rtl' align='right'> در ولیفندی هیپودروم، مسابقاتی از آوریل تا دسامبر وجود دارد. </p> |
|  contradiction | <p dir='rtl' align='right'> آیا کودکانی وجود دارند که نیاز به سرگرمی دارند؟ </p> | <p dir='rtl' align='right'> هیچ کودکی هرگز نمی خواهد سرگرم شود. </p> |
|  neutral | <p dir='rtl' align='right'> ما به سفرهایی رفته ایم که در نهرهایی شنا کرده ایم </p> | <p dir='rtl' align='right'> علاوه بر استحمام در نهرها ، ما به اسپا ها و سونا ها نیز رفته ایم. </p> |


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

To reproduce our numbers with all our baselines, try [`train_and_evaluate_qqp_baselines.sh`](scripts/train_and_evaluate_qqp_baselines.sh) script. 

 
### Reading Comprehension 
In this task, the goal is to generate a response to question and its accompanying context paragraph. 
Here are several examples: 

|  question | paragraph | answer |
| :---: | :---: | :---: |
|  بند ناف انسان به کجا وصل است؟ | ناف جایی قرار گرفته که در واقع بندناف در داخل رحم در آنجا به شکم جنین وصل بوده‌است. بندناف که جفت را به جنین متصل کرده بعد از تولد از نوزاد جدا می‌شود. برای جدا کردن بند ناف از دو پنس استفاده می‌کنند و بین آن دو را میبرند. پنس دیگری نزدیک شکم نوزاد قرار داده می‌شود که بعد از دو روز برداشته خواهد شد. بندناف باقی‌مانده طی ۱۵ روز خشک شده و می‌افتد و به جای آن اسکاری طبیعی به جای میماند. البته بر خلاف تصور عامه مردم شکل ناف در اثر بریدن بند ناف به وجود نمی‌آید و پیش از این در شکم مادر حالت ناف شکل گرفته‌است. شکل ناف در میان مردم مختلف متفاوت است و اندازه آن بین ۱.۵ تا ۲ سانتی‌متر است. تمام پستانداران جفت‌زیست ناف دارند. ناف در انسان‌ها به سادگی قابل مشاهده‌است. | جفت |
|  چرا خفاش در شب بیدار است؟ | بیشتر خفاش‌ها شب‌زی‌اند. آن‌ها در طول روز یا خوابند یا به پاکسازی بدن خود می‌پردازند و در هنگام شب به شکار می‌روند. ابزار مسیریابی و شکار خفاش‌ها در تاریکی تا دههٔ ۱۷۹۰ کاملاً ناشناخته بود تا اینکه کشیش و زیست شناس ایتالیایی لازارو اسپالانزانی به مجموعه آزمایش‌هایی بر روی خفاش‌های کور دست زد. این خفاش‌ها در یک اتاق کاملاً تاریک گذاشت و مسیر آن‌ها را با نخ‌های ابریشمی پُر پیچ و خم کرد. حتی در تاریکی مطلق هم شبکورها راه خود را در آن مسیر پر پیچ و خم پیدا کرده بودند به همین دلیل او نتیجه گرفت که ابزار راه‌یابی شبکورها چیزی غیر از چشمانشان است. | شکار |
|  قاره آمریکا در چه سالی کشف شد؟ | بیش از ده هزار سال است که انسان‌ها در قارهٔ آمریکا زندگی می‌کنند. قاره آمریکا توسط کریستف کلمب و در سال ۱۴۹۲ کشف شد اما او به اشتباه فکر کرد که آنجا هندوستان است اما مدت‌ها بعد آمریگو وسپوچی اعلام کرد که این قاره جدیدی است. اما تاریخ آمریکا به عنوان یک کشور مستقل به سال ۱۷۸۳ میلادی بازمی‌گردد که در آن آمریکا بر طبق معاهدهٔ پاریس به رسمیت شناخته گردید. | ۱۴۹۲ |
|  چه کسانی فدک را به پیامبر اعطا کردند؟ | یهودیان که از مسلمانان در جنگ‌های مختلفی شکست خورده بودند در جریان فتح فدک ناچار به صلح با محمد (پیامبر اسلام) شدند.<br/><br/>فدک در نزدیکی خیبر قرار داشت و با توجه به موقعیت استراتژیک خود نقطه اتکاء یهودیان حجاز به‌شمار می‌رفت. پس از آنکه سپاه اسلام، یهودیان را در «خیبر» و «وادی‌القری» و «تیما» شکست داد، برای پایان دادن به قدرت قوم یهود، سفیری به نام «محیط» به نزد سران فدک فرستادند. سران فدک صلح و تسلیم را بر جنگ ترجیح دادند و تعهد کردند که هر سال نیمی از محصولات فدک را در اختیار پیامبر قرار داده و از این به بعد زیر سلطه اسلام زندگی کنند. | یهودیان |
|  کدام دانشگاه ها رشته مترجمی زبان دارند؟ | رشته مترجمی زبان انگلیسی یکی از رشته‌ها در دانشگاه‌های ایران است که در آن کار ترجمه از زبان فارسی به انگلیسی و بالعکس به دانشجویان آموخته می‌شود. این رشته در سطح کاردانی کارشناسی کارشناسی ارشد و دکترا در بیشتر دانشگاه‌ها و موسسات آموزش عالی ایران وجود دارد. یک رشته نظری و عمومی با عنوان مطالعات ترجمه (که به یک زبان خاص مربوط نمی‌شود) در مقاطع بالاتر هم وجود دارد. | بیشتر دانشگاه‌ها و موسسات آموزش عالی ایران |
|  پنجاب مربوط کدام ولایت است؟ | پنجاب مرکز منطقۀ دایزنگی قدیم است، اما براساس تقسیمات اداری سال ۱۳۴۳ جزئی از ولایت بامیان شد. مرکز این ولسوالی هم پنجاب نام دارد. | بامیان |
|  چرا زمان پخش عزیزه تغییر کرد؟ | این سریال که پخش آن از 19 آبان ماه سه شنبه شب ها آغاز شده بود به دلیل جایگاه بدی که در رتبه بندی ها به دست آورده بود به روز شنبه منتقل شد شاید تاثیری در بهتر شدن رتبه بندی ها داشته باشد، نویسنده سریال نیز به همین دلیل تغییر کرده است. | جایگاه بد در رتبه بندی ها |
|  بیماری وبا از چه طریقی وارد بدن میشود؟ | وَبا، مرگامرگی یا کالِرا (به انگلیسی: Cholera) یک عفونت در روده باریک است که از طریق آب توسط باکتری ویبریو کلرا ایجاد می‌شود. این باکتری با نوشیدن آب آلوده یا خوردن ماهی نپخته یا خوردن صدف‌ها وارد بدن می‌شود. | نوشیدن آب آلوده یا خوردن ماهی نپخته یا خوردن صدف‌ها |
|  چرا فیلم رستاخیز اکران نشد؟ | فیلم رستاخیز در روز ۲۴ تیر ۱۳۹۴ با مجوز قانونی وزارت فرهنگ و ارشاد اسلامی به اکران عمومی درآمد اما ساعاتی پس از آن در پی مخالفت علما و مراجع با محتوای آن و به تصویر کشیدن چهره برخی از پرده سینماها به پایین کشیده شد. | مخالفت علما و مراجع با محتوای آن و به تصویر کشیدن چهره برخی |
|  چه چیزهایی در آزمایش خون مشخص می شود؟ | البته آزمایش خون هم می‌توان نشان دهد که شخص پیش از این به کرونا مبتلا بوده است یا نه. خوبی تست خون این است که مشخص می‌کند فرد در برابر این بیماری مصونیت پیدا کرده است یا نه. | شخص پیش از این به کرونا مبتلا بوده است یا نه |

To train a model, here is a sample script:
```bash
???
```

Training with the previously defined hyper-parameters yields the following results on the test set:
```
????
```

To reproduce our numbers with all our baselines, try [`train_and_evaluate_reading_comprehension_baselines.sh`](scripts/train_and_evaluate_reading_comprehension_baselines.sh) script.

 
 ### Multiple-Choice QA 
 Here the task is to pick a correct answer among 3-5 given candidate answers.
 Here are several examples: 

|  Question | Correct Answer | Candidate1 | Candidate2 | Candidate3 | Candidate4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  پایتخت کشور استرالیا کدام است؟ | 3 | ملبورن | سیدنی | کنبرا |  |
|  منظومه یا مجموعه عناصر و اجزائی که با هم کنش و واکنش و ارتباط متقابل دارند را چه می نامند؟ | 4 | نهاد | سازمان اجتماعی | گشتالت | سیستم |
|  کدام یک از موارد زیر جزء مراحل چهارگانه تصمیم گیری، نمی باشد؟ | 3 | تعریف و تشخیص مشکل | دستیابی به راح حل ها | هدایت و نظارت | اجرای تصممیم |
|  مفهوم کلی کدام بیت با سایر ابیات متفاوت است؟ | 4 | از خلاف آمد عادت بطلب کام که من          کسب جمعیت از آن زلف پریشان کردم | گفتم که بوی زلفت گمراه عالمم کرد        گفتا اگر بدانی هم اوت رهبر آید | زلف آشفته‌ی او موجب جمعیت ماست      چون چنین است پس آشفته‌ترش باید کرد | اگر به زلف دراز تو دست ما نرسد       گناه بخت پریشان و دست کوته ماست |
|  یک مجسمه، یک گلدان و یک ساعت را که از لحاظ حجم تقریبا به یک اندازه می‌باشند به چند صورت مختلف می‌توان دو بدو در کنار هم و روی یک کمد برای زینت اتاق قرار داد؟ | 1 | ۶ | ۴ | ۲ | ۸ |


To reproduce our baselines, try [`train_and_evaluate_multiple_choice_baselines.sh`](scripts/train_and_evaluate_multiple_choice_baselines.sh) script.
 
 ### Machine Translation 
Machine Translation of Persian/English is one of the few tasks that has received more work in the past few years. 
Unfortunately, most of the evaluation done for this task is often limited to few domains/datasets.    
Here we have compiled a collection of high-quality resources for Persian machine translation. 
Specifically, here is our collection of evaluation sets: 
 - **Quaran:** BVased on the existing translations of Quran.  
 - **Bible:** Based on the existing translations of Bible. 
 - **Mizan:** Parallel corpora constructed from human translations of literary masterpieces. 
 - **Global Voices:** 
 - **Quora queries:** the translation instance extracted from our [query paraphrasing task](#query-paraphrasing).  
 - **TE sentences:** the translation instances extracted from our [entailment task](#textual-entailment).    

Here are several examples: 
??


To reproduce our baselines, try [`train_and_evaluate_machine_translation_baselines.sh`](scripts/train_and_evaluate_machine_translation_baselines.sh) script.

 
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