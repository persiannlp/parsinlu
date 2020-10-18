export DATA_DIR=../data/multiple-choice

declare -a models=("xlm-roberta-base" "xlm-roberta-large" "akhooli/xlm-r-large-arabic-toxic" "akhooli/xlm-r-large-arabic-sent" "microsoft/MiniLM-L12-H384-uncased" "microsoft/Multilingual-MiniLM-L12-H384" "TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased" "m3hrdadfi/albert-fa-base-v2-clf-persiannews" "HooshvareLab/bert-fa-base-uncased-clf-persiannews")

# not used anymore
#"asafaya/bert-large-arabic" "akhooli/gpt2-small-arabic" "asafaya/bert-mini-arabic" "akhooli/xlm-r-large-arabic-sent" "salti/xlm-roberta-large-arabic_qa" "asafaya/bert-medium-arabic" "kuisailab/albert-base-arabic" "kuisailab/albert-xlarge-arabic" "xlm-mlm-tlm-xnli15-1024" "xlm-mlm-xnli15-1024" "xlm-mlm-17-1280" "xlm-mlm-100-1280"

for model in "${models[@]}"; do

  export TRAIN_BATCH_SIZE=16

  if [[ $model == *"large"* ]]; then
    export TRAIN_BATCH_SIZE=2
  fi

  python3.7 ../src/run_multiple_choice.py \
    --task_name multiple_choice_all \
    --data_dir $DATA_DIR \
    --model_name_or_path "${model}" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 64 \
    --output_dir "multiple_choice_models/${model}" \
    --save_steps -1 \
    --overwrite_output_dir
done
