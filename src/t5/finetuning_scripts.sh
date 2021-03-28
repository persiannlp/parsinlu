export PROJECT=...
export ZONE=...
export TPU_NAME=...
export BUCKET=...

########################## QQP #################################
#"small" "base" "large" "xxl"
declare -a sizes=("xl")

TASK=qqp
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
    --gin_param="utils.run.save_checkpoints_steps=1000" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
    --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done

########################## QQP-English #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  TASK=qqp_english
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
 python -m t5.models.mesh_transformer_main \
   --module_import="parsiglue_tasks" \
   --tpu="${TPU_NAME}" \
   --gcp_project="${PROJECT}" \
   --tpu_zone="${ZONE}" \
   --model_dir="${MODEL_DIR}" \
   --gin_file="dataset.gin" \
   --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
   --gin_param="utils.run.save_checkpoints_steps=1000" \
   --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
   --gin_param="MIXTURE_NAME = '${TASK}'" \
   --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
   --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
   --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
   --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
   --gin_location_prefix="multilingual_t5/gin/"

 # Run eval
 python -m t5.models.mesh_transformer_main \
   --module_import="parsiglue_tasks" \
   --tpu="${TPU_NAME}" \
   --gcp_project="${PROJECT}" \
   --tpu_zone="${ZONE}" \
   --model_dir="${MODEL_DIR}" \
   --gin_file="dataset.gin" \
   --gin_file="${MODEL_DIR}/operative_config.gin" \
   --gin_file="eval.gin" \
   --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
   --gin_param="MIXTURE_NAME = '${TASK}'" \
   --gin_param="utils.run.dataset_split = 'qqp_english_validation'" \
   --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
   --gin_param="utils.run.eval_checkpoint_step='all'" \
   --t5_tfds_data_dir="${BUCKET}/t5-tfds"

 python -m t5.models.mesh_transformer_main \
   --module_import="parsiglue_tasks" \
   --tpu="${TPU_NAME}" \
   --gcp_project="${PROJECT}" \
   --tpu_zone="${ZONE}" \
   --model_dir="${MODEL_DIR}" \
   --gin_file="dataset.gin" \
   --gin_file="${MODEL_DIR}/operative_config.gin" \
   --gin_file="eval.gin" \
   --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
   --gin_param="MIXTURE_NAME = '${TASK}'" \
   --gin_param="utils.run.dataset_split = 'qqp_english_test'" \
   --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
   --gin_param="utils.run.eval_checkpoint_step='all'" \
   --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  ## test on Persian
  TASK=qqp
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done

########################## QQP: English + Persian #################################
declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  TASK=qqp_mixture
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
 python -m t5.models.mesh_transformer_main \
   --module_import="parsiglue_tasks" \
   --tpu="${TPU_NAME}" \
   --gcp_project="${PROJECT}" \
   --tpu_zone="${ZONE}" \
   --model_dir="${MODEL_DIR}" \
   --gin_file="dataset.gin" \
   --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
   --gin_param="utils.run.save_checkpoints_steps=1000" \
   --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
   --gin_param="MIXTURE_NAME = '${TASK}'" \
   --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
   --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
   --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
   --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
   --gin_location_prefix="multilingual_t5/gin/"

 # Run eval
 python -m t5.models.mesh_transformer_main \
   --module_import="parsiglue_tasks" \
   --tpu="${TPU_NAME}" \
   --gcp_project="${PROJECT}" \
   --tpu_zone="${ZONE}" \
   --model_dir="${MODEL_DIR}" \
   --gin_file="dataset.gin" \
   --gin_file="${MODEL_DIR}/operative_config.gin" \
   --gin_file="eval.gin" \
   --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
   --gin_param="MIXTURE_NAME = '${TASK}'" \
   --gin_param="utils.run.dataset_split = 'qqp_english_validation'" \
   --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
   --gin_param="utils.run.eval_checkpoint_step='all'" \
   --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  ## test on Persian
  TASK=qqp
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done


########################## Multiple-Choice #################################

declare -a sizes=("small" "base" "large" "xl")

TASK=multiple_choice_str
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}_try2/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'valid'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ml'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_lit'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ck'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done

########################## Multiple-Choice: English #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  TASK=english_multiple_choice_arc_comqa_obqa
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
#    python -m t5.models.mesh_transformer_main \
#      --module_import="parsiglue_tasks" \
#      --tpu="${TPU_NAME}" \
#      --gcp_project="${PROJECT}" \
#      --tpu_zone="${ZONE}" \
#      --model_dir="${MODEL_DIR}" \
#      --gin_file="dataset.gin" \
#      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
#      --gin_param="utils.run.save_checkpoints_steps=1000" \
#      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
#      --gin_param="MIXTURE_NAME = '${TASK}'" \
#      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
#      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
#      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
#      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
#      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
#  python -m t5.models.mesh_transformer_main \
#    --module_import="parsiglue_tasks" \
#    --tpu="${TPU_NAME}" \
#    --gcp_project="${PROJECT}" \
#    --tpu_zone="${ZONE}" \
#    --model_dir="${MODEL_DIR}" \
#    --gin_file="dataset.gin" \
#    --gin_file="${MODEL_DIR}/operative_config.gin" \
#    --gin_file="eval.gin" \
#    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
#    --gin_param="MIXTURE_NAME = '${TASK}'" \
#    --gin_param="utils.run.dataset_split = 'dev'" \
#    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
#    --gin_param="utils.run.eval_checkpoint_step='all'" \
#    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

#  python -m t5.models.mesh_transformer_main \
#    --module_import="parsiglue_tasks" \
#    --tpu="${TPU_NAME}" \
#    --gcp_project="${PROJECT}" \
#    --tpu_zone="${ZONE}" \
#    --model_dir="${MODEL_DIR}" \
#    --gin_file="dataset.gin" \
#    --gin_file="${MODEL_DIR}/operative_config.gin" \
#    --gin_file="eval.gin" \
#    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
#    --gin_param="MIXTURE_NAME = '${TASK}'" \
#    --gin_param="utils.run.dataset_split = 'test'" \
#    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
#    --gin_param="utils.run.eval_checkpoint_step='all'" \
#    --t5_tfds_data_dir="${BUCKET}/t5-tfds"


  TASK=multiple_choice_str
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ml'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_lit'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"


    python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ck'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done



########################## Multiple-Choice: English + Persian #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  TASK=multiple_choice_mixture
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  TASK=multiple_choice_str
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'valid'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"


  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ml'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_lit'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"


    python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_ck'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done



########################## machine translation: en->fa #################################


declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=200000

for SIZE in "${sizes[@]}"; do
  TASK=translation_combined_en_fa
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

done


########################## machine translation: fa->en #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=200000

for SIZE in "${sizes[@]}"; do
  TASK=translation_combined_fa_en
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

done



########################## machine translation: ar->en #################################

export TPU_NAME=danielk-tpu-europe-west4-a-v3-8-3-new
export PROJECT=ai2-tpu
export ZONE=europe-west4-a
export BUCKET=gs://danielk-files/mt5-models

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=200000

for SIZE in "${sizes[@]}"; do
  TASK=arabic_english_opus100
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  TASK=translation_combined_fa_en
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

done


########################## machine translation: ar->en + fa -> en  #################################


declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=200000

for SIZE in "${sizes[@]}"; do
  TASK=translation_x_en_mixture
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  TASK=translation_combined_fa_en
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"


    python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

done


########################## entailment #################################

declare -a sizes=("small" "base" "large" "xl")

TASK=parsiglue_entailment
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_farstail'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_natural'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_translation'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done

########################## entailment: snli english #################################

#declare -a sizes=("small" "base" "large" "xl")
declare -a sizes=("xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=snli_entailment
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  # evaluate on ParsiGLUE
  TASK=parsiglue_entailment
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_farstail'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_natural'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_translation'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done



########################## entailment: english + persian #################################

#declare -a sizes=("small" "base" "large" "xl")
declare -a sizes=("large")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=entailment_mixture
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
    python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  # evaluate on ParsiGLUE
  TASK=parsiglue_entailment
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_farstail'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_natural'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'test_translation'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done


########################## reading comprehension: Persian #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=parsiglue_readingcomprehension
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  # evaluate on ParsiGLUE
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'eval'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done


########################## reading comprehension: squad1_1 #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=squad1_1
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
#    python -m t5.models.mesh_transformer_main \
#      --module_import="parsiglue_tasks" \
#      --tpu="${TPU_NAME}" \
#      --gcp_project="${PROJECT}" \
#      --tpu_zone="${ZONE}" \
#      --model_dir="${MODEL_DIR}" \
#      --gin_file="dataset.gin" \
#      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
#      --gin_param="utils.run.save_checkpoints_steps=1000" \
#      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
#      --gin_param="MIXTURE_NAME = '${TASK}'" \
#      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
#      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
#      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
#      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
#      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  # evaluate on ParsiGLUE
  TASK=parsiglue_readingcomprehension
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'eval'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done


########################## reading comprehension: English + Persian #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=reading_com_mixture
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  # evaluate on ParsiGLUE
  TASK=parsiglue_readingcomprehension
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'eval'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

done

########################## sentiment analysis #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=parsiglue_sentiment
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'merged_dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'movie_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'food_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done

########################## sentiment analysis: English #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=restuarant_reviews_english_dataset
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"


  TASK=parsiglue_sentiment
  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'merged_dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'movie_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'food_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done




########################## sentiment analysis: English + Persian #################################

declare -a sizes=("small" "base" "large" "xl")

PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000

for SIZE in "${sizes[@]}"; do
  PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
  TASK=sentiment_mixture
  MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"

  # Run fine-tuning
    python -m t5.models.mesh_transformer_main \
      --module_import="parsiglue_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=1000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"


  TASK=parsiglue_sentiment
  # Run eval
  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'merged_dev'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'movie_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  python -m t5.models.mesh_transformer_main \
    --module_import="parsiglue_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="utils.run.dataset_split = 'food_test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done
