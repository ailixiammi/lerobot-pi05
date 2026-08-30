#!/usr/bin/env bash
set -Eeuo pipefail

VLA_ROOT="${VLA_ROOT:-/path/to/vla}"
REPO="${LEROBOT_REPO:-${VLA_ROOT}/code/lerobot}"
DATA_ROOT="${DATA_ROOT:-${VLA_ROOT}/data/libero_v30}"
POLICY_ROOT="${POLICY_ROOT:-${VLA_ROOT}/models/pi05_base}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-${VLA_ROOT}/models/paligemma-3b-pt-224}"
RUN_ROOT="${RUN_ROOT:-${VLA_ROOT}/outputs/lerobot}"
LOG_ROOT="${LOG_ROOT:-${VLA_ROOT}/logs/lerobot}"
RUN_NAME=${1:-pi05_libero_smoke_1step_20260816}
OUTPUT_DIR=${RUN_ROOT}/${RUN_NAME}
LOG_FILE=${LOG_ROOT}/${RUN_NAME}.console.log

for required in \
  "${DATA_ROOT}/meta/info.json" \
  "${POLICY_ROOT}/config.json" \
  "${POLICY_ROOT}/model.safetensors" \
  "${POLICY_ROOT}/policy_preprocessor.json" \
  "${TOKENIZER_ROOT}/tokenizer.model" \
  "${TOKENIZER_ROOT}/tokenizer.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${REPO}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${VLA_ROOT}/data/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_LINK_MODE=copy

TRAIN_CMD=(
  "${REPO}/.venv/bin/lerobot-train"
  --dataset.repo_id=physical-intelligence/libero_v30
  --dataset.root="${DATA_ROOT}"
  --dataset.image_transforms.enable=false
  --policy.path="${POLICY_ROOT}"
  --policy.tokenizer_name="${TOKENIZER_ROOT}"
  --policy.input_features=null
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.train_expert_only=true
  --policy.push_to_hub=false
  --accelerator.mixed_precision=bf16
  '--rename_map={"image":"observation.images.base_0_rgb","wrist_image":"observation.images.left_wrist_0_rgb","state":"observation.state","actions":"action"}'
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}"
  --batch_size=1
  --steps=1
  --num_workers=0
  --log_freq=1
  --env_eval_freq=0
  --wandb.enable=false
  --save_checkpoint=false
)

echo "run_name=${RUN_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
echo "start_time=$(date --iso-8601=seconds)"

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "end_time=$(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
echo "launcher_exit_code=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
