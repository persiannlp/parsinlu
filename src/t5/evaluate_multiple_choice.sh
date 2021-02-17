declare -a sizes=("small" "base" "large" "xl")
declare -a splits=("test_ck" "test_ml" "test_lit" "valid" "dev")
declare -a models=("english_multiple_choice_arc_comqa_obqa" "multiple_choice_str")
for model in "${models[@]}"; do
  for size in "${sizes[@]}"; do
    for split in "${splits[@]}"; do
      python3.7 evaluate_predictions.py \
          --eval_path=mt5-models/${model}/${size}/${split}_eval \
          --eval_metric=multiple-choice --bucket_name=danielk-files --dump
    done
  done
done