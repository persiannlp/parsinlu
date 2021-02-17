declare -a sizes=("small" "base" "large" "xl")
declare -a splits=("dev" "test")
#declare -a models=("translation_combined_fa_en" "translation_combined_en_fa" "arabic_english_opus100")
#declare -a models=("translation_combined_en_fa" "arabic_english_opus100")
declare -a models=("arabic_english_opus100")
for model in "${models[@]}"; do
  for split in "${splits[@]}"; do
    for size in "${sizes[@]}"; do
      echo "----------------"
      echo "* model: ${model}"
      echo "* size: ${size}"
      echo "* split: ${split}"
      python3.7 evaluate_predictions.py \
          --eval_path=mt5-models/${model}/${size}/${split}_eval \
          --eval_metric=translation --bucket_name=danielk-files --dump
    done
  done
done


