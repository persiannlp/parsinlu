export DATA_DIR=../data/multiple-choice

#declare -a models=("bert-base-multilingual-cased" "bert-base-multilingual-uncased" "xlm-roberta-base" "xlm-roberta-large" "xlm-mlm-tlm-xnli15-1024" "xlm-mlm-xnli15-1024" "xlm-mlm-17-1280" "xlm-mlm-100-1280" "m3hrdadfi/albert-fa-base-v2-clf-persiannews" "HooshvareLab/bert-fa-base-uncased-clf-persiannews" "asafaya/bert-large-arabic" "akhooli/gpt2-small-arabic" "asafaya/bert-mini-arabic" "akhooli/xlm-r-large-arabic-sent" "salti/xlm-roberta-large-arabic_qa" "asafaya/bert-medium-arabic" "kuisailab/albert-base-arabic" "kuisailab/albert-xlarge-arabic")
declare -a models=("bert-base-multilingual-cased")

for model in "${models[@]}"; do

python3.7 ../src/run_multiple_choice.py \
  --task_name multiple_choice_all \
  --data_dir $DATA_DIR \
  --model_name_or_path "${model}" \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir "entailment_model_${model}" \
  --save_steps -1
done
