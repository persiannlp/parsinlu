export DATA_DIR=../data/qqp

python ../src/run_text_classification.py \
  --data_dir $DATA_DIR \
  --task_name qqp \
  --model_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_qqp/ \
  --save_steps -1


