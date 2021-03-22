# ParsiNLU
ParsiNLU is a comprehensive suit of high-level NLP tasks for Persian language. 
This suit contains 6 different key NLP tasks --- *Reading Comprehension*, *Multiple-Choice Question-Answering*, 
*Textual Entailment*, *Sentiment Analysis*, *Query Paraphrasing* and *Machine Translation*. 

These challenges are collected by expert annotators native in Persian language and from various sources. 
If you'd like to see additional details on the dataset and how we collected, please check out [our publication](#citation).    
  
## Getting the data 
You can find the data under the [`data/`](data) directory.  

<!---
## Leaderboard
On our [leaderboard page](https://parsiglue.com) we host the state-of-art scores for each challenge task.  
 -->
 
## Examples and Baselines  
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

|  Split | en | fa |
| :---: | :---: | :---: |
|  Quran | Praise be to Allah, the Cherisher and Sustainer of the worlds; |<p dir='rtl' align='right'> ستایش خدای را که پروردگار جهانیان است. </p>|
|  Quran | This is the Book; in it is guidance sure, without doubt, to those who fear Allah; |<p dir='rtl' align='right'> این کتاب که هیچ شک در آن نیست، راهنمای پرهیزگاران است. </p>|
|  Quran | When they meet those who believe, they say: "We believe;" but when they are alone with their evil ones, they say: "We are really with you: We (were) only jesting." |  <p dir='rtl' align='right'>و چون به اهل ایمان برسند گویند: ما ایمان آوردیم؛ و وقتی با شیاطین خود خلوت کنند گویند: ما با شماییم، جز این نیست که (مؤمنان را) مسخره می‌کنیم.</p>|
|  Quran | Who believe in the Unseen, are steadfast in prayer, and spend out of what We have provided for them; |  <p dir='rtl' align='right'>آن کسانی که به جهان غیب ایمان آرند و نماز به پا دارند و از هر چه روزیشان کردیم به فقیران انفاق کنند.</p>|
|  Bible | And God called the dry land Earth; and the gathering together of the waters called he Seas: and God saw that it was good. |<p dir='rtl' align='right'>  و خدا خشکی را زمین نامید و اجتماع آبها رادریا نامید. و خدا دید که نیکوست. </p>|
|  Bible | And God said, Let the earth bring forth grass, the herb yielding seed, and the fruit tree yielding fruit after his kind, whose seed is in itself, upon the earth: and it was so. | <p dir='rtl' align='right'> و خداگفت: «زمین نباتات برویاند، علفی که تخم بیاوردو درخت میوه‌ای که موافق جنس خود میوه آوردکه تخمش در آن باشد، بر روی زمین.» و چنین شد. </p>|
|  Bible | And the earth brought forth grass, and herb yielding seed after his kind, and the tree yielding fruit, whose seed was in itself, after his kind: and God saw that it was good. | <p dir='rtl' align='right'> و زمین نباتات را رویانید، علفی که موافق جنس خود تخم آورد و درخت میوه داری که تخمش در آن، موافق جنس خود باشد. و خدادید که نیکوست.</p>|
|  Mizan | But Proportion has a sister, less smiling, more formidable, a Goddess even now engaged | <p dir='rtl' align='right'>اما تناسب امور خواهری دارد، نه این چنین متبسم، رعب آور‌تر، ایزد بانویی که حتی در این لحظه مشغول است. </p>|
|  Mizan | At Hyde Park Corner on a tub she stands preaching; |<p dir='rtl' align='right'> در هاید پارک کرنر بر گلدانی ایستاده موعظه می‌کند؛ </p>|
|  Mizan | shrouds herself in white and walks penitentially disguised as brotherly love through factories and parliaments; offers help, but desires power; | <p dir='rtl' align='right'> پیچیده در دایی سفید به نشان توبه با لباس مبدل عشق برادرانه در کارخانه‌ها و مجالس قانونگذاری راه می‌رود؛ پیشنهاد کمک می‌کند، اما طالب قدرت است.</p>|
|  QQP | What turns people off about Quora? | <p dir='rtl' align='right'> چه چیزی مردم را از Quora دور می کند؟</p>|
|  QQP | Is there a way to turn off the "Invite People to Join Quora' option on Quora? |  <p dir='rtl' align='right'>آیا راهی برای خاموش کردن گزینه "دعوت از مردم برای پیوستن به Quora" در Quora وجود دارد؟</p>|
|  QQP | What were the books studied by aiims topper 2016? |  <p dir='rtl' align='right'>کتابهایی که توسط aiims topper ۲۰۱۶ مورد مطالعه قرار گرفته چه بود؟</p>|
|  QQP | What books should I study for my PG entrance in AIIMS? |  <p dir='rtl' align='right'>برای ورود PG من در AIIMS چه کتابهایی باید مطالعه کنم؟</p>|
|  QQP | Which website is good for downloading Android (.apk) files? |<p dir='rtl' align='right'>  کدام وب سایت برای دانلود پرونده های Android (.apk) مناسب است؟ </p>|
|  QQP | Android Application Development: Which software is used to develop APK files? |  <p dir='rtl' align='right'>توسعه برنامه Android: از کدام نرم افزار برای توسعه فایل های APK استفاده می شود؟</p>|


To downloading the data, take a look at the [this](https://github.com/persiannlp/parsinlu/tree/master/data/translation/translation_combined_en_fa) and [that](https://github.com/persiannlp/parsinlu/tree/master/data/translation/translation_combined_fa_en) directories. 

All the baselines here use T5. Take a look at [this section](#reproducing-mt5-baselines) for more description on that. 

 
 ### Sentiment Analysis
 Our aspect-based sentiment analysis task includes three sub-tasks: 1) detecting the overall sentiment of a review/document, 2) extracting aspects toward which a sentiment is expressed, and 3) detecting the sentiment polarity of extracted aspects. Our annotation scheme is mainly inspired by the [`Sem-Eval 2014 Task 4`](https://www.aclweb.org/anthology/S14-2004/), ABSA scheme, with minor adjustments. Sentiment scores are chosen from `(very negative, negative, neutral, positive, very positive, mixed/borderline)`. 
 
 So far, we have annotated documents from `food & beverages` ([`Digikala`](https://www.digikala.com/main/food-beverage/)) and `movie review` ([`Tiwall`](https://www.tiwall.com/)) domains. We have predefined list of aspects for each domain. In the following, we have listed some examples from our dataset:

| Domain  | Review | Sentiment  | (Aspect, Sentiment) |
| :---: | ------------- | :---: | :---: |
| Food & beverages  | <p dir='rtl' align='right'>خیلی خیلی کادوی جذابیه هم بسته بندی شیک هم شکلات خوشمزه و قلبی شکل خصوصا که پاکت هم داره</p> | Very positive <img width=400/> | (بسته بندی، خیلی مثبت) <br>(طعم، مثبت) <img width=500/>|
| Food & beverages  | <p dir='rtl' align='right'>در شگفت انگیز به قیمت خیلی پایین خریدم ولی به نظرم ارزش نداره و طعم خاصی جز شکر نداره</p> | Negative | (ارزش خرید، منفی)<br> (طعم، منفی) |
| Movie review  |<p dir='rtl' align='right'>در جشنواره متاسفانه نتونستم ببینم ولی دیشب در اکران فیلم های جشنواره فجر در پردیس چارسو موفق به دیدن فیلم شدم. چه فیلم خوبی از فضای بصری زیبا و چشم نواز، تا بازی فوق العاده حامد بهداد.....</p> | Positive | (صحنه، مثبت)<br> (بازی، خیلی مثبت) |
| Movie review  |<p dir='rtl' align='right'>فیلمی بسیار ضعیف، علی الخصوص در زمینه ی تدوین و فیلم نامه پر از شعار زدگی، کلیشه و اغراق آمیز!!! واقعا خانم درخشنده توی این فیلم تنزل فاحشی پیدا کردن. بعد اصلا معلوم نیست اون زن دوم اون وسط چی میگه، از بس که شخصیت پردازی ضعیفه!</p> | Very negative | (بازی، خیلی منفی) <br>(داستان، خیلی منفی) <br>(کارگردانی، خیلی منفی) |
| Movie review | <p dir='rtl' align='right'>فیلم از فضای نقد اجتماعی و سیاسی تهی است...یه قصه غیر قابل باور که هیجان خاصی نداشت...ریتم فیلم قابل قبول بود...الناز شاکردوست هم خیلی فراتر از انتظار بود...نمره 5 از 10</p> | Mixed/borderline | (داستان، منفی)<br> (بازی، خیلی مثبت) |


To reproduce our numbers with all our baselines, try [`train_and_evaluate_sentiment_analysis_baselines.sh`](scripts/train_and_evaluate_sentiment_analysis_baselines.sh) script.


## Using the finetuned models using the HuggingFace🤗 library 

Our models are deployed on [HuggingFace's model hub](https://huggingface.co/models).
You can our list of models in [this page](https://huggingface.co/persiannlp).  
Each model readme contains descriptions on how to use it. 

## Reproducing mT5 baselines 
To use mT5 baselines you need the followings: 
 - Convert all the data into train/test/dev splits organized in TSV format (first column: input, 2nd column: output). For example, see [the translation data](https://github.com/persiannlp/parsinlu/tree/master/data/translation/translation_combined_fa_en). We have a [script](https://github.com/persiannlp/parsinlu/blob/master/src/t5/create_t5_data.py) for this conversion. 
 - A Cloud storage: best to use [Google cloud buckets](https://cloud.google.com/storage/docs/json_api/v1/buckets) to store your train/eval/dev data. 
 - A TPU: for the experiments in this work we used v3-8 TPUs. You can get such TPUs for free from [the TFRC program](https://www.tensorflow.org/tfrc) (thanks, Google!!).  
 - A [virtual machine](https://cloud.google.com/compute) to run the experiments: create one CPU machine, in the same zone/region as your TPU.  
 - [Install T5](https://github.com/google-research/text-to-text-transfer-transformer/#installation) and run the [file-tuning scripts](https://github.com/persiannlp/parsinlu/blob/master/src/t5/finetuning_scripts.sh) on your virtual machine. 
 - After your obtain the predictions, use our evaluation scripts in [this directory](https://github.com/persiannlp/parsinlu/tree/master/src/t5). 

 **Note**: If you don't have access to TPUs, you should be able to use the HuggingFace library to train mT5 on your GPU machine. We don't have the scripts for that, but you should be ale to follow [the examples here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq). 


## FAQ 
**I have GPU on my machine by `n_gpu` is shown as `0`. Where is the problem?** Check out [this thread](https://github.com/pytorch/pytorch/issues/15612).  

## Citation 
If you find this work useful please cite the following work: 
```bibtex 
@article{2020parsiglue,
    title={{ParsiNLU:} A Suite of Language Understanding Challenges for Persian},
    author={Daniel Khashabi, Arman Cohan, Siamak Shakeri, Pedram Hosseini, Pouya Pezeshkpour, Malihe Alikhani, Moin Aminnaseri, Marzieh Bitaab, Faeze Brahman, Sarik Ghazarian, Mozhdeh Gheini, Arman Kabiri, Rabeeh Karimi Mahabadi, Omid Memarrast, Ahmadreza Mosallanezhad, Erfan Noury, Shahab Raji, Mohammad Sadegh Rasooli, Sepideh Sadeghi, Erfan Sadeqi Azer, Niloofar Safi Samghabadi, Mahsa Shafaei, Saber Sheybani, Ali Tazarv, Yadollah Yaghoobzadeh},
    journal={arXiv},
    year={2020}
}
```

If you use the translatin task, please cite the following work as well: 
```bibtex
@article{kashefi2018mizan,
  title={MIZAN: a large persian-english parallel corpus},
  author={Kashefi, Omid},
  journal={arXiv preprint arXiv:1801.02107},
  year={2018}
}
``` 
