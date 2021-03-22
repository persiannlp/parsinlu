import json


# qqp
def convert_qqp_file(file):
    outf = open(file.replace(".jsonl", ".tsv"), "+w")
    with open(file) as f:
        for line in f.readlines():
            json_line = json.loads(line)
            q1 = json_line['q1']
            q2 = json_line['q2']
            label = json_line['label']
            # category = json_line['category']
            outf.write(f"{q1} <sep> {q2}\t{label}\n")


# convert_qqp_file("../../data/qqp/train.jsonl")
# convert_qqp_file("../../data/qqp/dev.jsonl")
# convert_qqp_file("../../data/qqp/test.jsonl")

# qqp in English
def convert_qqp_english_file():
    from datasets import load_dataset
    dataset = load_dataset('glue', 'qqp')
    print(dataset)
    for split in ['train', 'test', 'validation']:
        outf = open(f"../../data/qqp/qqp_english_{split}.tsv", "+w")
        subdataset = dataset[split]
        for instance in subdataset:
            q1 = instance['question1'].replace("\n", "\\n").replace("\r", " ")
            q2 = instance['question2'].replace("\n", "\\n").replace("\r", " ")
            label = instance['label']
            outf.write(f"{q1} <sep> {q2}\t{label}\n")


# convert_qqp_english_file()

# multiple-choice
def convert_multiple_choice_questions(file):
    outf = open(file.replace(".jsonl", ".tsv"), "+w")
    with open(file, 'r') as infile:
        for i, line in enumerate(infile.readlines()):
            json_line = json.loads(line)
            question = json_line['question']
            answer = int(json_line['answer'])
            candidates = json_line['candidates']
            candidates = [c for c in candidates if len(c.strip()) > 0]
            print(answer)
            print(candidates)
            answer_str = candidates[answer - 1].replace("\n", "\\n").replace("\r", " ").replace("\"", "").replace("\t",
                                                                                                                  "").replace(
                "'", "").replace("n\\", "").replace("\\", "")
            candidates = [f"<sep> {c}" for (i, c) in enumerate(candidates)]
            candidates = " ".join(candidates)
            candidates = candidates.replace("\n", " ").replace("\r", " ").replace("\"", "").replace("\t", "").replace(
                "'", "").replace("n\\", "").replace("\\", "")
            question = question.replace("\n", " ").replace("\r", " ").replace("\"", "").replace("\t", "").replace("'",
                                                                                                                  "").replace(
                "n\\", "").replace("\\", "")
            outf.write(f"{question} {candidates}\t{answer_str}\n")


# convert_multiple_choice_questions("../../data/multiple-choice/train.jsonl")
# convert_multiple_choice_questions("../../data/multiple-choice/test_ck.jsonl")
# convert_multiple_choice_questions("../../data/multiple-choice/test_lit.jsonl")
# convert_multiple_choice_questions("../../data/multiple-choice/test_ml.jsonl")
# convert_multiple_choice_questions("../../data/multiple-choice/valid.jsonl")

separators = ['(A)', '(B)', '(C)', '(D)', '(E)']


def create_multiple_chice_english_data(dir):
    def read_file(file):
        outfile = open("../../data/multiple-choice/english/english_multiple_choice_arc_comqa_obqa/" + file, "a")
        with open(dir + file) as f:
            for line in f:
                line = line.replace("\n", "")
                line_split = line.split("\t")
                question = line_split[0]
                question = question.replace("\\n", " ")
                for s in separators:
                    question = question.replace(s, "<sep>")
                answer = line_split[1]
                outfile.write(question + "\t" + answer + "\n")

    read_file("train.tsv")
    read_file("dev.tsv")
    if dir != "commonsenseqa":
        read_file("test.tsv")


# create_multiple_chice_english_data("../../data/multiple-choice/english/arc_easy/")
# create_multiple_chice_english_data("../../data/multiple-choice/english/arc_hard/")
# create_multiple_chice_english_data("../../data/multiple-choice/english/commonsenseqa/")
# create_multiple_chice_english_data("../../data/multiple-choice/english/openbookqa/")

# machine translation
def convert_quaran_data():
    files = [
        'en.ahmedali.txt',
        'en.ahmedraza.txt',
        'en.arberry.txt',
        'en.daryabadi.txt',
        'en.hilali.txt',
        'en.itani.txt',
        'en.maududi.txt',
        'en.mubarakpuri.txt',
        'en.yusufali.txt',
        'fa.ansarian.norm.txt',
        'fa.ayati.norm.txt',
        'fa.bahrampour.norm.txt',
        'fa.fooladvand.norm.txt',
        'fa.gharaati.norm.txt',
        'fa.ghomshei.norm.txt',
        'fa.khorramshahi.norm.txt',
        'fa.makarem.norm.txt',
        'fa.moezzi.norm.txt',
        'fa.sadeqi.norm.txt'
    ]

    fa_lines_per_file = {
        'combined': []
    }
    en_lines_per_file = {
        'combined': []
    }

    en_ref = 'en.yusufali.txt'
    fa_ref = 'fa.fooladvand.norm.txt'

    for file in files:
        with open("../../data/translation/quran/" + file, "r") as f:
            all_lines = list(f.readlines())
            if 'en' in file:
                en_lines_per_file[file] = all_lines
                if len(en_lines_per_file['combined']) == 0:
                    en_lines_per_file['combined'] = all_lines
                else:
                    for i, _ in enumerate(all_lines):
                        en_lines_per_file['combined'][i] += "///" + all_lines[i]
            else:
                fa_lines_per_file[file] = all_lines
                if len(fa_lines_per_file['combined']) == 0:
                    fa_lines_per_file['combined'] = all_lines

                for i, _ in enumerate(all_lines):
                    fa_lines_per_file['combined'][i] += "///" + all_lines[i]

    outf_enfa = open("../../data/translation/quran/quran_en_fa.tsv", "+w")
    outf_faen = open("../../data/translation/quran/quran_fa_en.tsv", "+w")

    for (en1, fa1, en_all, fa_all) in zip(en_lines_per_file[en_ref], fa_lines_per_file[fa_ref],
                                          en_lines_per_file['combined'], fa_lines_per_file['combined']):
        fa1 = fa1.replace('\n', '').replace('\t', '')
        en1 = en1.replace('\n', '').replace('\t', '')
        fa_all = fa_all.replace('\n', '').replace('\t', ' ')
        en_all = en_all.replace('\n', '').replace('\t', ' ')
        outf_enfa.write(f"{en1}\t{fa_all}\n")
        outf_faen.write(f"{fa1}\t{en_all}\n")


def convert_tep_data():
    with open("../../data/translation/tep/TEP.en-fa.en", 'r') as f:
        lines_en = list(f.readlines())

    with open("../../data/translation/tep/TEP.en-fa.fa", 'r') as f:
        lines_fa = list(f.readlines())

    outf_enfa = open("../../data/translation/tep/tep_en_fa.tsv", "+w")
    outf_faen = open("../../data/translation/tep/tep_fa_en.tsv", "+w")
    for (en, fa) in zip(lines_en, lines_fa):
        fa = fa.replace('\n', '').replace('\t', ' ')
        en = en.replace('\n', '').replace('\t', ' ')
        outf_enfa.write(f"{en}\t{fa}\n")
        outf_faen.write(f"{fa}\t{en}\n")


def convert_bible_data():
    import xml.etree.ElementTree as ET

    def read_content(lang):
        lines = []
        root = ET.fromstring(open('../../data/translation/bible/' + lang + '.xml').read())
        with open(lang + '.txt', 'w', encoding='utf-8') as out:
            for n in root.iter('seg'):
                lines.append(n.text.strip())
        return lines

    lines_en = read_content('English')
    lines_fa = read_content('Farsi')

    outf_enfa = open("../../data/translation/bible/bible_en_fa.tsv", "+w")
    outf_faen = open("../../data/translation/bible/bible_fa_en.tsv", "+w")

    for (en, fa) in zip(lines_en, lines_fa):
        fa = fa.replace('\n', '').replace('\t', ' ')
        en = en.replace('\n', '').replace('\t', ' ')
        outf_enfa.write(f"{en}\t{fa}\n")
        outf_faen.write(f"{fa}\t{en}\n")


def convert_mizan_data():
    def convert_file(file):
        with open(file + ".fa") as f:
            lines_fa = list(f.readlines())
        with open(file + ".en") as f:
            lines_en = list(f.readlines())

        split = file.split("/")[-1]
        outf_enfa = open(f"../../data/translation/mizan/mizan_{split}_en_fa.tsv", "+w")
        outf_faen = open(f"../../data/translation/mizan/mizan_{split}_fa_en.tsv", "+w")

        for (en, fa) in zip(lines_en, lines_fa):
            fa = fa.replace('\n', '').replace('\t', ' ')
            en = en.replace('\n', '').replace('\t', ' ')
            outf_enfa.write(f"{en}\t{fa}\n")
            outf_faen.write(f"{fa}\t{en}\n")

    convert_file('../../data/translation/mizan/dev')
    convert_file('../../data/translation/mizan/test')
    convert_file('../../data/translation/mizan/train')


def convert_global_voices():
    with open("../../data/translation/global_voices/GlobalVoices.en-fa.norm.fa") as f:
        lines_fa = list(f.readlines())

    with open("../../data/translation/global_voices/GlobalVoices.en-fa.en") as f:
        lines_en = list(f.readlines())

    outf_enfa = open(f"../../data/translation/global_voices/global_voices_en_fa.tsv", "+w")
    outf_faen = open(f"../../data/translation/global_voices/global_voices_fa_en.tsv", "+w")

    for (en, fa) in zip(lines_en, lines_fa):
        fa = fa.replace('\n', '').replace('\t', ' ')
        en = en.replace('\n', '').replace('\t', ' ')
        outf_enfa.write(f"{en}\t{fa}\n")
        outf_faen.write(f"{fa}\t{en}\n")


def convert_queries_data():
    outfile_fa_en_train = open("../../data/qqp/qqp_train_fa_en.tsv", "w")
    outfile_fa_en_dev = open("../../data/qqp/qqp_dev_fa_en.tsv", "w")
    outfile_fa_en_test = open("../../data/qqp/qqp_test_fa_en.tsv", "w")

    outfile_en_fa_train = open("../../data/qqp/qqp_train_en_fa.tsv", "w")
    outfile_en_fa_dev = open("../../data/qqp/qqp_dev_en_fa.tsv", "w")
    outfile_en_fa_test = open("../../data/qqp/qqp_test_en_fa.tsv", "w")

    with open("../../data/qqp/QQP-all-final.tsv") as f:
        for line in f.readlines():
            line_split = line.split("\t")
            q1_en = line_split[3].replace('\t', ' ')
            q2_en = line_split[4].replace('\t', ' ')
            q1_fa = line_split[8].replace('\t', ' ')
            q2_fa = line_split[9].replace('\t', ' ')
            split = line_split[11]
            if len(q1_en) > 5 and len(q1_fa) > 5:
                if split == "train":
                    outfile_fa_en_train.write(f"{q1_fa}\t{q1_en}\n")
                    outfile_fa_en_train.write(f"{q2_fa}\t{q2_en}\n")
                elif split == "test":
                    outfile_fa_en_test.write(f"{q1_fa}\t{q1_en}\n")
                    outfile_fa_en_test.write(f"{q2_fa}\t{q2_en}\n")
                else:
                    outfile_fa_en_dev.write(f"{q1_fa}\t{q1_en}\n")
                    outfile_fa_en_dev.write(f"{q2_fa}\t{q2_en}\n")

                if split == "train":
                    outfile_en_fa_train.write(f"{q1_en}\t{q1_fa}\n")
                    outfile_en_fa_train.write(f"{q2_en}\t{q2_fa}\n")
                elif split == "test":
                    outfile_en_fa_test.write(f"{q1_en}\t{q1_fa}\n")
                    outfile_en_fa_test.write(f"{q2_en}\t{q2_fa}\n")
                else:
                    outfile_en_fa_dev.write(f"{q1_en}\t{q1_fa}\n")
                    outfile_en_fa_dev.write(f"{q2_en}\t{q2_fa}\n")


# convert_queries_data()
# convert_tep_data()
# convert_bible_data()
# convert_mizan_data()
# convert_global_voices()
# convert_quaran_data()


def combine_tsv_files(outfile_name, files):
    combined_file = open(outfile_name, "w")
    for file in files:
        short_name = file.split("/")[-1].replace(".tsv", "")
        with open(file, "r") as f:
            for line in f.readlines():
                combined_file.write(line.replace("\n", "") + f"\t{short_name}\n")


def combine_translation_datasets():
    test_sets = [
        "../../data/translation/bible/bible_fa_en.tsv",
        "../../data/translation/mizan/mizan_test_fa_en.tsv",
        "../../data/translation/quran/quran_fa_en.tsv",
        "../../data/translation/queries/qqp_test_fa_en.tsv",
    ]

    combine_tsv_files("../../data/translation/translation_combined_fa_en/test.tsv", test_sets)

    dev_sets = [
        "../../data/translation/mizan/mizan_dev_fa_en.tsv",
        "../../data/translation/queries/qqp_dev_fa_en.tsv"
    ]

    combine_tsv_files("../../data/translation/translation_combined_fa_en/dev.tsv", dev_sets)

    train_sets = [
        "../../data/translation/global_voices/global_voices_fa_en.tsv",
        "../../data/translation/mizan/mizan_train_fa_en.tsv",
        "../../data/translation/tep/tep_fa_en.tsv",
        "../../data/translation/queries/qqp_train_fa_en.tsv"
    ]
    combine_tsv_files("../../data/translation/translation_combined_fa_en/train.tsv", train_sets)

    test_sets = [
        "../../data/translation/bible/bible_en_fa.tsv",
        "../../data/translation/mizan/mizan_test_en_fa.tsv",
        "../../data/translation/quran/quran_en_fa.tsv",
        "../../data/translation/queries/qqp_train_en_fa.tsv"
    ]
    combine_tsv_files("../../data/translation/translation_combined_en_fa/test.tsv", test_sets)

    dev_sets = [
        "../../data/translation/mizan/mizan_dev_en_fa.tsv",
        "../../data/translation/queries/qqp_dev_en_fa.tsv"
    ]
    combine_tsv_files("../../data/translation/translation_combined_en_fa/dev.tsv", dev_sets)

    train_sets = [
        "../../data/translation/global_voices/global_voices_en_fa.tsv",
        "../../data/translation/mizan/mizan_train_en_fa.tsv",
        "../../data/translation/tep/tep_en_fa.tsv",
        "../../data/translation/queries/qqp_test_en_fa.tsv"
    ]
    combine_tsv_files("../../data/translation/translation_combined_en_fa/train.tsv", train_sets)


# combine_translation_datasets()

def convert_opus_data():
    def combine_file_pair(arabic_file, english_file):
        with open(arabic_file, 'r') as f:
            arabic_lines = list(f.readlines())
        with open(english_file, 'r') as f:
            english_lines = list(f.readlines())

        outfile = open(english_file.replace(".en", ".txt"), 'w')

        assert len(english_lines) == len(arabic_lines)

        for aline, eline in zip(arabic_lines, english_lines):
            outfile.write(
                aline.replace("\n", "").replace("\t", "") + "\t" + eline.replace("\n", "").replace("\t", "") + "\n")

    combine_file_pair(
        "../../data/translation/arabic_english_opus100/opus.ar-en-dev.ar",
        "../../data/translation/arabic_english_opus100/opus.ar-en-dev.en",
    )
    combine_file_pair(
        "../../data/translation/arabic_english_opus100/opus.ar-en-test.ar",
        "../../data/translation/arabic_english_opus100/opus.ar-en-test.en",
    )
    combine_file_pair(
        "../../data/translation/arabic_english_opus100/opus.ar-en-train.ar",
        "../../data/translation/arabic_english_opus100/opus.ar-en-train.en",
    )


# convert_opus_data()

# textual entailment
import csv


def convert_entailment_data():
    def convert_parsiglue_file(file, outfilename, category=None):
        outfile = open(outfilename, "w")
        with open(file) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                sent1 = row[1].replace("\t", "").replace("\n", "")
                sent2 = row[2].replace("\t", "").replace("\n", "")
                label = row[3].replace("\t", "").replace("\n", "")
                cat = row[4].replace("\t", "").replace("\n", "")
                if category:
                    if category in cat:
                        outfile.write(f"{sent1}<sep>{sent2}\t{label}\n")
                else:
                    outfile.write(f"{sent1}<sep>{sent2}\t{label}\n")

    convert_parsiglue_file(
        '../../data/entailment/dev.csv',
        '../../data/entailment/merged_with_farstail/dev.tsv')

    convert_parsiglue_file(
        '../../data/entailment/train.csv',
        '../../data/entailment/merged_with_farstail/train.tsv')

    convert_parsiglue_file(
        '../../data/entailment/test.csv',
        '../../data/entailment/merged_with_farstail/test_natural.tsv', 'natural')

    convert_parsiglue_file(
        '../../data/entailment/test.csv',
        '../../data/entailment/merged_with_farstail/test_translation.tsv', 'translation')

    def convert_farstail_file(file, outfilename):
        outfile = open(outfilename, "w")
        with open(file) as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                print(row)
                print(file)
                sent1 = row[0].replace("\t", "").replace("\n", "")
                sent2 = row[1].replace("\t", "").replace("\n", "")
                label = row[2].replace("\t", "").replace("\n", "")
                outfile.write(f"{sent1}<sep>{sent2}\t{label}\n")

    convert_farstail_file(
        '../../data/entailment/farstail/dev.csv',
        '../../data/entailment/merged_with_farstail/dev_farstail.tsv')

    convert_farstail_file(
        '../../data/entailment/farstail/train.csv',
        '../../data/entailment/merged_with_farstail/train_farstail.tsv')

    convert_farstail_file(
        '../../data/entailment/farstail/test.csv',
        '../../data/entailment/merged_with_farstail/test_farstail.tsv')


# convert_entailment_data()

def convert_snli_file(file):
    outfile = open(file.replace('.jsonl', '.tsv'), 'w')
    with open(file) as f:
        for line in f.readlines():
            json_line = json.loads(line)
            sent1 = json_line['sentence1'].replace("\t", "").replace("\n", "")
            sent2 = json_line['sentence2'].replace("\t", "").replace("\n", "")
            label = json_line['gold_label']
            if label == 'neutral':
                label = 'n'
            elif label == 'entailment':
                label = 'e'
            elif label == 'contradiction':
                label = 'c'
            else:
                continue
                print(f'WARNING: invalid label: {label}')
            outfile.write(f'{sent1}<sep>{sent2}\t{label}\n')


# convert_snli_file('../../data/entailment/snli/snli_1.0/snli_1.0_dev.jsonl')
# convert_snli_file('../../data/entailment/snli/snli_1.0/snli_1.0_train.jsonl')
# convert_snli_file('../../data/entailment/snli/snli_1.0/snli_1.0_test.jsonl')

sentiment_labels = {
    '-3': 'no sentiment expressed',
    '-2': 'very negative',
    '-1': 'negative',
    '0': 'neutral',
    '1': 'positive',
    '2': 'very positive',
    '3': 'mixed',
}


def convert_sentiment_data(file):
    io_outfile = open(file.replace(".jsonl", ".tsv"), "+w")
    with open(file) as f:
        for line in f.readlines():
            json_line = json.loads(line.replace("\n", "").replace("'", '"'))
            review = json_line['review']
            question = json_line['question']
            label = json_line['label']
            label = sentiment_labels[label]
            io_outfile.write(f"{review} <sep> {question}\t{label}\n")


# convert_sentiment_data('../../data/sentiment-analysis/ABSA_Dataset_train.jsonl')
# convert_sentiment_data('../../data/sentiment-analysis/food_test.jsonl')
# convert_sentiment_data('../../data/sentiment-analysis/movie_test.jsonl')
# convert_sentiment_data('../../data/sentiment-analysis/merged_dev.jsonl')

def create_English_sentiment_data(file):
    pass


def convert_reading_comprehension():
    def convert_file(infile, is_train):
        outfile = open(infile.replace(".jsonl", ".tsv"), "+w")
        with open(infile) as f:
            for line in f.readlines():
                json_line = json.loads(line)
                question = json_line['question'].replace("\n", " ").replace("\t", " ")
                # url = json_line['url']
                passage = json_line['passage'].replace("\n", " ").replace("\t", " ")
                answers = json.dumps(json_line['answers']).replace("\n", " ").replace("\t", " ")
                if is_train:
                    answers = json_line['answers'][0][1].replace("\n", " ").replace("\t", " ")
                outfile.write(f"{question} \\n {passage}\t{answers}\n")

    convert_file("../../data/reading_comprehension/eval.jsonl", False)
    convert_file("../../data/reading_comprehension/dev.jsonl", False)
    convert_file("../../data/reading_comprehension/train.jsonl", True)

# convert_reading_comprehension()
