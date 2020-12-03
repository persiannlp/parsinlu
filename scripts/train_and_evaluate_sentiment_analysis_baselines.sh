export DATA_DIR=../data/sentiment-analysis

# first, clean tokenizer caches
rm ${DATA_DIR}/cached_*

# Preparing data
python3.7 -u ../src/pre-processor/sentiment-analysis/data_builder.py --domain food --input_dir $DATA_DIR --output_dir $DATA_DIR
python3.7 -u ../src/pre-processor/sentiment-analysis/data_builder.py --domain movie --input_dir $DATA_DIR --output_dir $DATA_DIR
python3.7 -u ../src/pre-processor/sentiment-analysis/train_set_merger.py --domains food movie --input_dir $DATA_DIR --output_dir $DATA_DIR

# Training and evaluating models
declare -a models=("TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")

# Hyper-parameter tuning
declare -a learning_rates=(3e-5 5e-5)
declare -a num_train_epochs=(3 7)


# for model in "${models[@]}"; do
# python3.7 -u ../src/run_text_classification.py \
#   --data_dir $DATA_DIR \
#   --task_name sentiment \
#   --model_name_or_path "${model}" \
#   --do_train \
#   --do_eval \
#   --learning_rate 5e-5 \
#   --num_train_epochs 3 \
#   --max_seq_length 64 \
#   --output_dir "sentiment_model/${model}" \
#   --save_steps -1
# done


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
          --task_name sentiment \
          --model_name_or_path "${model}" \
          --do_train \
          --do_eval \
          --learning_rate "${learning_rate[@]}" \
          --num_train_epochs "${num_train_epoch[@]}" \
          --max_seq_length 64 \
          --per_gpu_train_batch_size "${batch_size[@]}" \
          --per_gpu_eval_batch_size "${batch_size[@]}" \
          --output_dir "sentiment_model/${model}" \
          --output_dir "sentiment_model/${model}_batch_size=${batch_size}_learning_rate=${learning_rate}_learning_rate=${learning_rate}_num_train_epoch=${num_train_epoch}" \
          --save_steps -1
      done
    done
  done
done

exit 0  # stop here; evaluation (following scrpts) requires manual intervention

## After selecting your best checkpoints based on the dev sets, run the following script
# Update this following line with the path to your best checkpoints
eval_dir="sentiment_model/TurkuNLP/wikibert-base-fa-cased_batch_size=8_learning_rate=5e-5_learning_rate=5e-5_num_train_epoch=3"

# notice --eval_on_test which would force the code to use the test set for evaluation
python ../src/run_text_classification.py \
  --task_name sentiment \
  --data_dir $DATA_DIR \
  --model_name_or_path ${eval_dir} \
  --tokenizer_name  ${eval_dir} \
  --do_eval \
  --eval_on_test \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 64 \
  --output_dir "${eval_dir}"