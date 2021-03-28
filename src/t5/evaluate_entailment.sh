#declare -a sizes=("small" "base" "large" "xl")
declare -a sizes=("large")
declare -a splits=("dev" "test_natural" "test_translation" "test_farstail")
declare -a models=("entailment_mixture")
for model in "${models[@]}"; do
  for size in "${sizes[@]}"; do
    for split in "${splits[@]}"; do
      python3.7 evaluate_predictions.py \
          --eval_path=mt5-models/${model}/${size}/${split}_eval \
          --eval_metric=accuracy --bucket_name=danielk-files --dump
    done
  done
done