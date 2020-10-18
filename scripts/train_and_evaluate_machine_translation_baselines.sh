export DATA_DIR=../data/translation/quran
export PYTHONPATH="../src/":"${PYTHONPATH}"

#declare -a models=("HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased" "xlm-roberta-base" "xlm-roberta-large" "xlm-mlm-tlm-xnli15-1024" "xlm-mlm-xnli15-1024" "xlm-mlm-17-1280" "xlm-mlm-100-1280" "m3hrdadfi/albert-fa-base-v2-clf-persiannews" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "asafaya/bert-large-arabic" "akhooli/gpt2-small-arabic" "asafaya/bert-mini-arabic" "akhooli/xlm-r-large-arabic-sent" "salti/xlm-roberta-large-arabic_qa" "asafaya/bert-medium-arabic" "kuisailab/albert-base-arabic" "kuisailab/albert-xlarge-arabic")

#sshleifer/tiny-mbart --> did not work for translation; bug?

cp ../data/translation/quran/en.ahmedali.txt ../data/translation/quran/train.source
cp ../data/translation/quran/en.ahmedraza.txt ../data/translation/quran/val.source
cp ../data/translation/quran/en.arberry.txt ../data/translation/quran/test.source

cp ../data/translation/quran/fa.ansarian.norm.txt ../data/translation/quran/train.target
cp ../data/translation/quran/fa.ayati.norm.txt ../data/translation/quran/val.target
cp ../data/translation/quran/fa.bahrampour.norm.txt ../data/translation/quran/test.target

#export model=facebook/mbart-large-cc25
export model=t5-small

python3.7 ../src/run_seq2seq.py \
    --data_dir $DATA_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=translation_${model}_results \
    --num_train_epochs 1 \
    --model_name_or_path ${model} \
    --learning_rate=3e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --sortish_sampler
