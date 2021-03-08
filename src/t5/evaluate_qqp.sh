declare -a sizes=("small" "base" "large" "xl")
declare -a splits=("dev")
declare -a models=("qqp_mixture")
for model in "${models[@]}"; do
  for size in "${sizes[@]}"; do
    for split in "${splits[@]}"; do
      python3.7 evaluate_predictions.py \
          --eval_path=mt5-models/${model}/${size}/${split}_eval \
          --eval_metric=accuracy --bucket_name=danielk-files --dump
    done
  done
done