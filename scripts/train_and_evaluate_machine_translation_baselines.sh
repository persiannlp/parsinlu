#export DATA_DIR=../data/multiple-choice
export DATA_DIR=${PWD}/data/translation/quran
export PYTHONPATH="../src/":"${PYTHONPATH}"

#declare -a models=("HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased" "xlm-roberta-base" "xlm-roberta-large" "xlm-mlm-tlm-xnli15-1024" "xlm-mlm-xnli15-1024" "xlm-mlm-17-1280" "xlm-mlm-100-1280" "m3hrdadfi/albert-fa-base-v2-clf-persiannews" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "asafaya/bert-large-arabic" "akhooli/gpt2-small-arabic" "asafaya/bert-mini-arabic" "akhooli/xlm-r-large-arabic-sent" "salti/xlm-roberta-large-arabic_qa" "asafaya/bert-medium-arabic" "kuisailab/albert-base-arabic" "kuisailab/albert-xlarge-arabic")
#
#for model in "${models[@]}"; do
#
#python3.7 ../src/run_multiple_choice.py \
#  --task_name multiple_choice_all \
#  --data_dir $DATA_DIR \
#  --model_name_or_path "${model}" \
#  --do_train \
#  --do_eval \
#  --learning_rate 5e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 64 \
#  --output_dir "multiple_choice_models/${model}" \
#  --save_steps -1
#done

python3.7 ../src/run_seq2seq.py \
    --data_dir $DATA_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=xsum_results \
    --num_train_epochs 1 \
    --model_name_or_path bert-base-multilingual-uncased \
    --learning_rate=3e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --sortish_sampler
