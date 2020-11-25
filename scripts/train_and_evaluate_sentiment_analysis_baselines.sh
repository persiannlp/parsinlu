export DATA_DIR=../data/sentiment-analysis

# Preparing data
python3.7 -u ../src/pre-processor/Sentiment-analysis/SA_Data_Builder.py --domain food --input_dir $DATA_DIR --output_dir $DATA_DIR
python3.7 -u ../src/pre-processor/Sentiment-analysis/SA_Data_Builder.py --domain movie --input_dir $DATA_DIR --output_dir $DATA_DIR
python3.7 -u ../src/pre-processor/Sentiment-analysis/TrainSetMerger.py --domains food movie --input_dir $DATA_DIR --output_dir $DATA_DIR

# Training and evaluating models
declare -a models=("TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased")

for model in "${models[@]}"; do
python3.7 -u ../src/run_text_classification.py \
  --data_dir $DATA_DIR \
  --task_name sentiment \
  --model_name_or_path "${model}" \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 64 \
  --output_dir "sentiment_model/${model}" \
  --save_steps -1
done
