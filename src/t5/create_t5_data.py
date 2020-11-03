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
    import csv
    outf = open(file.replace(".csv", ".tsv"), "+w")
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            print("----")
            print(i)
            if i == 0:
                continue
            print(row)
            question = row[1]
            answer = int(row[2])
            print(answer)
            candidates = row[3:]
            candidates = [c for c in candidates if len(c.strip()) > 0]
            print(candidates)
            answer_str = candidates[answer-1].replace("\n", "\\n").replace("\r", " ")
            candidates = [f"<sep> {c}" for (i, c) in enumerate(candidates)]
            candidates = " ".join(candidates)
            candidates = candidates.replace("\n", "\\n").replace("\r", " ")
            question = question.replace("\n", "\\n").replace("\r", " ")
            outf.write(f"{question} {candidates}\t{answer_str}\n")


# convert_multiple_choice_questions("../../data/multiple-choice/train.csv")
# convert_multiple_choice_questions("../../data/multiple-choice/test_ck.csv")
# convert_multiple_choice_questions("../../data/multiple-choice/test_lit.csv")
# convert_multiple_choice_questions("../../data/multiple-choice/test_ml.csv")
# convert_multiple_choice_questions("../../data/multiple-choice/valid.csv")

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
        fa1 = fa1.replace('\n', '')
        en1 = en1.replace('\n', '')
        fa_all = fa_all.replace('\n', '')
        en_all = en_all.replace('\n', '')
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
        fa = fa.replace('\n', '')
        en = en.replace('\n', '')
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
        fa = fa.replace('\n', '')
        en = en.replace('\n', '')
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
            fa = fa.replace('\n', '')
            en = en.replace('\n', '')
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
        fa = fa.replace('\n', '')
        en = en.replace('\n', '')
        outf_enfa.write(f"{en}\t{fa}\n")
        outf_faen.write(f"{fa}\t{en}\n")


def convert_queries_data():
    pass


def convert_entailment_data():
    pass


# convert_tep_data()
# convert_bible_data()
# convert_mizan_data()
# convert_global_voices()
# convert_quaran_data()

# textual entailment
