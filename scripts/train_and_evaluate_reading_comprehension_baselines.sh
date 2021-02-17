# not currently functional since we have not added the training data for reading-comprehension. 
# should be fixed in a couple of weeks. 

#export DATA_DIR=../data/multiple-choice
export DATA_DIR=${PWD}/wmt_en_ro
export PYTHONPATH="../src/":"${PYTHONPATH}"

python3.7 ../src/run_seq2seq.py \
    --data_dir $DATA_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=xsum_results \
    --num_train_epochs 1 \
    --model_name_or_path facebook/bart-base \
    --learning_rate=3e-5 \
    --gpus 0 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --sortish_sampler
