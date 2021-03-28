export DATA_DIR=../data/reading_comprehension

# first, clean tokenizer caches
rm ${DATA_DIR}/cached_*

declare -a models=("TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")
declare -a learning_rates=(3e-5 5e-5)
declare -a num_train_epochs=(3 7)

for model in "${models[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for num_train_epoch in "${num_train_epochs[@]}"; do
        python ../src/run_squad.py \
            --model_type bert \
            --model_name_or_path "${model}" \
            --do_train \
            --do_eval \
            --train_file $DATA_DIR/train.json \
            --predict_file $DATA_DIR/dev.json \
            --learning_rate "${learning_rate[@]}" \
            --num_train_epochs "${num_train_epoch[@]}" \
            --max_seq_length 384 \
            --doc_stride 128 \
            --output_dir "reading_comprehension_model/${model}_learning_rate=${learning_rate}_learning_rate=${learning_rate}_num_train_epoch=${num_train_epoch}" \
            --per_gpu_eval_batch_size=256  \
            --per_gpu_train_batch_size=4   \
            --save_steps 5000
    done
  done
done