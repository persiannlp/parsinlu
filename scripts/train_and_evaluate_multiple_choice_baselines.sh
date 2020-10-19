# first, clean tokenizer caches
rm ../data/multiple-choice/cached_*

export DATA_DIR=../data/multiple-choice

declare -a models=("TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")

#"xlm-roberta-base" "xlm-roberta-large"

declare -a learning_rates=(3e-5 4e-5 5e-5)
declare -a num_train_epochs=(3 5 7 9)

for model in "${models[@]}"; do

  declare -a batch_sizes=(8 16)
  if [[ $model == *"large"* ]]; then
    declare -a batch_sizes=(1 2)
  fi

  for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for num_train_epoch in "${num_train_epochs[@]}"; do

        python3.7 ../src/run_multiple_choice.py \
          --task_name multiple_choice_all \
          --data_dir $DATA_DIR \
          --model_name_or_path "${model}" \
          --tokenizer_name  "${model}" \
          --do_train \
          --do_eval \
          --per_device_train_batch_size "${batch_size[@]}" \
          --learning_rate "${learning_rate[@]}" \
          --num_train_epochs "${num_train_epoch[@]}" \
          --max_seq_length 64 \
          --output_dir "multiple_choice_models/${model}_batch_size=${batch_size}_learning_rate=${learning_rate}_learning_rate=${learning_rate}_num_train_epoch=${num_train_epoch}" \
          --save_steps -1 \
          --overwrite_output_dir
      done
    done
  done
done
