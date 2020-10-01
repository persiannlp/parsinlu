# ParsiGLUE
 - intro and what it is 
 - where to download the data 


## Baselines  
 - Download the data from the ParsiGLUE repo (https://github.com/persiannlp/parsiglue) and include it in the `data/` directory.  
 - See the relevant section on how to train each classifier  
    * Textual entailment 
    * Query Paraphrasing 
    * Reading Comprehension 
   

 ### Textual Entailment 
 TODO 
 
 ### Query Paraphrasing 
 QQP is the task of detecting whether two given questions are paraphrases of each other or not. 

 This example code fine-tunes mBERT (multi-lingual BERT) on the this task. 
 It runs in xxx mins on a single tesla V100 16GB. 

```bash 
export DATA_DIR=/path/to/XNLI
export DATA_DIR=/Users/danielk/ideaProjects/parsiglue-baselines/data/qqp

python run_text_classification.py \
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
```

--per_device_train_batch_size 8 \

Training with the previously defined hyper-parameters yields the following results on the test set:
 
```
acc = ?
```
 
 ### Reading Comprehension 
 TODO 
 
 ### Multiple-Choice QA 
 TODO 
 
 ### Machine Translation 
 TODO 
 
 ### Sentiment Analysis 
TODO 

## Citation 
If you find this work useful please cite the following work: 
```bibtex 
@article{2020parsiglue,
    title={},
    author={},
    journal={arXiv},
    year={2020}
}
```