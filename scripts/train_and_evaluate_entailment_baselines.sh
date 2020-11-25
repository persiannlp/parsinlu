export DATA_DIR=../data/entailment/merged_with_farstail

# first, clean tokenizer caches
rm ${DATA_DIR}/cached_*

declare -a models=("TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")

declare -a learning_rates=(3e-5 5e-5)
declare -a num_train_epochs=(3 7)

for model in "${models[@]}"; do
  declare -a batch_sizes=(8 16)
  if [[ $model == *"large"* ]]; then
    declare -a batch_sizes=(1 2)
  fi

  for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for num_train_epoch in "${num_train_epochs[@]}"; do

        python3.7 ../src/run_text_classification.py \
          --data_dir $DATA_DIR \
          --task_name entailment \
          --model_name_or_path "${model}" \
          --do_train \
          --do_eval \
          --learning_rate "${learning_rate[@]}" \
          --num_train_epochs "${num_train_epoch[@]}" \
          --max_seq_length 64 \
          --per_gpu_train_batch_size "${batch_size[@]}" \
          --per_gpu_eval_batch_size "${batch_size[@]}" \
          --output_dir "entailment_model/${model}" \
          --output_dir "entailment_model/${model}_batch_size=${batch_size}_learning_rate=${learning_rate}_learning_rate=${learning_rate}_num_train_epoch=${num_train_epoch}" \
          --save_steps -1
      done
    done
  done
done

exit 0  # stop here; evaluation (following scrpts) requires manual intervention

## After selecting your best checkpoints based on the dev sets, run the following script
# Update this following line with the path to your best checkpoints
eval_dir="entailment_model/TurkuNLP/wikibert-base-fa-cased_batch_size=16_learning_rate=5e-5_learning_rate=5e-5_num_train_epoch=3"

# notice --eval_on_test which would force the code to use the test set for evaluation
python ../src/run_text_classification.py \
  --task_name entailment \
  --data_dir $DATA_DIR \
  --model_name_or_path ${eval_dir} \
  --tokenizer_name  ${eval_dir} \
  --do_eval \
  --eval_on_test \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 64 \
  --output_dir "${eval_dir}"

