export DATA_DIR=../data/sentiment-analysis

#declare -a models=("microsoft/MiniLM-L12-H384-uncased" "microsoft/Multilingual-MiniLM-L12-H384" "TurkuNLP/wikibert-base-fa-cased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-fa-base-uncased" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "HooshvareLab/bert-base-parsbert-uncased" "bert-base-multilingual-cased" "bert-base-multilingual-uncased" "xlm-roberta-base" "xlm-roberta-large" "xlm-mlm-tlm-xnli15-1024" "xlm-mlm-xnli15-1024" "xlm-mlm-17-1280" "xlm-mlm-100-1280" "m3hrdadfi/albert-fa-base-v2-clf-persiannews" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "asafaya/bert-large-arabic" "akhooli/gpt2-small-arabic" "asafaya/bert-mini-arabic" "akhooli/xlm-r-large-arabic-sent" "salti/xlm-roberta-large-arabic_qa" "asafaya/bert-medium-arabic" "kuisailab/albert-base-arabic" "kuisailab/albert-xlarge-arabic")

declare -a models=("HooshvareLab/bert-base-parsbert-uncased")
#declare -a models=("microsoft/MiniLM-L12-H384-uncased")

for model in "${models[@]}"; do

python3.7 -u ../src/run_text_classification.py \
  --data_dir $DATA_DIR \
  --task_name sentiment \
  --model_name_or_path "${model}" \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --max_seq_length 64 \
  --output_dir "sentiment_model/${model}" \
  --save_steps -1
done
