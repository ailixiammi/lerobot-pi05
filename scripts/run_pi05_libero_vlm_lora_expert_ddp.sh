#!/usr/bin/env bash
set -Eeuo pipefail

VLA_ROOT="${VLA_ROOT:-/path/to/vla}"
REPO="${LEROBOT_REPO:-${VLA_ROOT}/code/lerobot}"
DATA_ROOT="${DATA_ROOT:-${VLA_ROOT}/data/libero_v30}"
BASE_POLICY_ROOT="${BASE_POLICY_ROOT:-${VLA_ROOT}/models/pi05_base}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-${VLA_ROOT}/models/paligemma-3b-pt-224}"
DATASET_CACHE="${DATASET_CACHE:-${VLA_ROOT}/cache/huggingface-datasets}"
RUN_ROOT="${RUN_ROOT:-${VLA_ROOT}/outputs/lerobot}"
LOG_ROOT="${LOG_ROOT:-${VLA_ROOT}/logs/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
TRAIN_BIN="${TRAIN_BIN:-${REPO}/.venv/bin/lerobot-train}"

EXP_NAME=pi05_libero_vlm_lora_expert_full
GLOBAL_BATCH_SIZE=8
STEPS=10000
SAVE_FREQ=1000
GRADIENT_ACCUMULATION_STEPS=1
LORA_R=16
LORA_ALPHA=16
NUM_WORKERS=4
SCOPE=language
RESUME=false
RESUME_CONFIG=
OVERWRITE=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_pi05_libero_vlm_lora_expert_ddp.sh [options]

  --exp-name NAME
  --base-policy PATH         Full PI0.5 checkpoint used as the frozen starting point.
  --scope SCOPE              language (default) or language-vision.
  --global-batch-size N       Effective global batch across all GPUs and accumulation steps.
  --steps N                   Total optimizer updates.
  --save-freq N               Save every N optimizer updates.
  --gradient-accumulation-steps N
  --lora-r N
  --lora-alpha N
  --num-workers N             DataLoader workers per rank (may be zero).
  --resume [TRAIN_CONFIG]     Resume from TRAIN_CONFIG, or this run's checkpoints/last.
  --overwrite                 Remove an existing non-resume output directory.
  --dry-run                   Validate and print the torchrun/DDP command without executing it.
  -h, --help
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_positive_integer() {
  local name=$1 value=$2
  [[ ${value} =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got '${value}'."
}

require_nonnegative_integer() {
  local name=$1 value=$2
  [[ ${value} =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got '${value}'."
}

while (($#)); do
  case "$1" in
    --exp-name) (($# >= 2)) || die "--exp-name requires a value"; EXP_NAME=$2; shift 2 ;;
    --exp-name=*) EXP_NAME=${1#*=}; shift ;;
    --base-policy) (($# >= 2)) || die "--base-policy requires a value"; BASE_POLICY_ROOT=$2; shift 2 ;;
    --base-policy=*) BASE_POLICY_ROOT=${1#*=}; shift ;;
    --scope) (($# >= 2)) || die "--scope requires a value"; SCOPE=$2; shift 2 ;;
    --scope=*) SCOPE=${1#*=}; shift ;;
    --global-batch-size) (($# >= 2)) || die "--global-batch-size requires a value"; GLOBAL_BATCH_SIZE=$2; shift 2 ;;
    --global-batch-size=*) GLOBAL_BATCH_SIZE=${1#*=}; shift ;;
    --steps) (($# >= 2)) || die "--steps requires a value"; STEPS=$2; shift 2 ;;
    --steps=*) STEPS=${1#*=}; shift ;;
    --save-freq) (($# >= 2)) || die "--save-freq requires a value"; SAVE_FREQ=$2; shift 2 ;;
    --save-freq=*) SAVE_FREQ=${1#*=}; shift ;;
    --gradient-accumulation-steps)
      (($# >= 2)) || die "--gradient-accumulation-steps requires a value"
      GRADIENT_ACCUMULATION_STEPS=$2
      shift 2
      ;;
    --gradient-accumulation-steps=*) GRADIENT_ACCUMULATION_STEPS=${1#*=}; shift ;;
    --lora-r) (($# >= 2)) || die "--lora-r requires a value"; LORA_R=$2; shift 2 ;;
    --lora-r=*) LORA_R=${1#*=}; shift ;;
    --lora-alpha) (($# >= 2)) || die "--lora-alpha requires a value"; LORA_ALPHA=$2; shift 2 ;;
    --lora-alpha=*) LORA_ALPHA=${1#*=}; shift ;;
    --num-workers) (($# >= 2)) || die "--num-workers requires a value"; NUM_WORKERS=$2; shift 2 ;;
    --num-workers=*) NUM_WORKERS=${1#*=}; shift ;;
    --resume)
      RESUME=true
      if (($# >= 2)) && [[ $2 != --* ]]; then RESUME_CONFIG=$2; shift 2; else shift; fi
      ;;
    --resume=*) RESUME=true; RESUME_CONFIG=${1#*=}; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ ${EXP_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  die "--exp-name must contain only letters, digits, dot, underscore, and dash."
[[ -n ${BASE_POLICY_ROOT} ]] || die "--base-policy must not be empty."
require_positive_integer --global-batch-size "${GLOBAL_BATCH_SIZE}"
require_positive_integer --steps "${STEPS}"
require_positive_integer --save-freq "${SAVE_FREQ}"
require_positive_integer --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
require_positive_integer --lora-r "${LORA_R}"
require_positive_integer --lora-alpha "${LORA_ALPHA}"
require_nonnegative_integer --num-workers "${NUM_WORKERS}"
[[ ${RESUME} == false || ${OVERWRITE} == false ]] || die "--resume and --overwrite are mutually exclusive."

case ${SCOPE} in
  language)
    TARGET_MODULES='.*\.paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj'
    FREEZE_VISION_ENCODER=true
    TARGET_DESCRIPTION='Gemma-2B language backbone q/v projections (18 layers)'
    ;;
  language-vision)
    TARGET_MODULES='.*\.paligemma_with_expert\.paligemma\.model\.(language_model\.layers\.[0-9]+|vision_tower\.vision_model\.encoder\.layers\.[0-9]+)\.self_attn\.(q|v)_proj'
    FREEZE_VISION_ENCODER=false
    TARGET_DESCRIPTION='Gemma-2B q/v (18 layers) + SigLIP q/v (27 layers)'
    ;;
  *) die "--scope must be 'language' or 'language-vision', got '${SCOPE}'." ;;
esac
FULL_TRAINING_MODULES='["model.paligemma_with_expert.gemma_expert","model.action_in_proj","model.action_out_proj","model.time_mlp_in","model.time_mlp_out"]'

if [[ -n ${CUDA_VISIBLE_DEVICES+x} ]]; then
  visible=${CUDA_VISIBLE_DEVICES//[[:space:]]/}
  [[ -n ${visible} ]] || die "CUDA_VISIBLE_DEVICES is set but empty."
  IFS=',' read -r -a cuda_devices <<< "${visible}"
  GPU_COUNT=${#cuda_devices[@]}
  for device in "${cuda_devices[@]}"; do
    [[ -n ${device} ]] || die "invalid CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
  done
else
  [[ -x ${PYTHON_BIN} ]] || die "Python executable not found: ${PYTHON_BIN}"
  GPU_COUNT=$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')
fi
require_positive_integer gpu-count "${GPU_COUNT}"

batch_divisor=$((GPU_COUNT * GRADIENT_ACCUMULATION_STEPS))
if ((GLOBAL_BATCH_SIZE % batch_divisor != 0)); then
  lower=$((GLOBAL_BATCH_SIZE / batch_divisor * batch_divisor))
  upper=$((((GLOBAL_BATCH_SIZE + batch_divisor - 1) / batch_divisor) * batch_divisor))
  if ((lower > 0)); then nearby="${lower} or ${upper}"; else nearby="${upper}"; fi
  die "global batch ${GLOBAL_BATCH_SIZE} is not divisible by GPU count ${GPU_COUNT} x gradient accumulation ${GRADIENT_ACCUMULATION_STEPS} (${batch_divisor}); nearby legal values: ${nearby}."
fi
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / batch_divisor))
EFFECTIVE_GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS))
EXPECTED_MICRO_STEPS=$((STEPS * GRADIENT_ACCUMULATION_STEPS))

OUTPUT_DIR=${RUN_ROOT}/${EXP_NAME}
LOG_FILE=${LOG_ROOT}/${EXP_NAME}.console.log
if [[ ${RESUME} == true && -z ${RESUME_CONFIG} ]]; then
  RESUME_CONFIG=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
fi

TRAIN_ARGS=(
  --dataset.repo_id=physical-intelligence/libero_v30
  --dataset.root="${DATA_ROOT}"
  --dataset.image_transforms.enable=false
  --policy.tokenizer_name="${TOKENIZER_ROOT}"
  --policy.input_features=null
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.train_expert_only=false
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
  --policy.push_to_hub=false
  --peft.method_type=LORA
  --peft.r="${LORA_R}"
  --peft.lora_alpha="${LORA_ALPHA}"
  --peft.target_modules="${TARGET_MODULES}"
  --peft.full_training_modules="${FULL_TRAINING_MODULES}"
  --accelerator.mixed_precision=bf16
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION_STEPS}"
  --parallelism.dp_replicate="${GPU_COUNT}"
  --parallelism.dp_shard=1
  '--rename_map={"image":"observation.images.base_0_rgb","wrist_image":"observation.images.left_wrist_0_rgb","state":"observation.state","actions":"action"}'
  --output_dir="${OUTPUT_DIR}"
  --job_name="${EXP_NAME}"
  --batch_size="${PER_DEVICE_BATCH_SIZE}"
  --steps="${STEPS}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --checkpoint_format=safetensors
  --num_workers="${NUM_WORKERS}"
  --log_freq=1
  --env_eval_freq=0
  --wandb.enable=false
  --save_checkpoint_to_hub=false
)
if [[ ${RESUME} == true ]]; then
  TRAIN_ARGS+=(--resume=true --config_path="${RESUME_CONFIG}")
else
  TRAIN_ARGS+=(--policy.path="${BASE_POLICY_ROOT}")
fi
CMD=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${GPU_COUNT}" "${TRAIN_BIN}" "${TRAIN_ARGS[@]}")

echo "world size: ${GPU_COUNT}"
echo "DDP topology: dp_replicate=${GPU_COUNT}, dp_shard=1"
echo "distributed strategy: DDP (VLM LoRA + fully trainable Action Expert)"
echo "base policy: ${BASE_POLICY_ROOT}"
echo "LoRA scope: ${SCOPE}"
echo "LoRA targets: ${TARGET_DESCRIPTION}"
echo "Action Expert: full training (Gemma Expert + action input/output + time MLP)"
echo "LoRA rank/alpha: ${LORA_R}/${LORA_ALPHA}"
echo "global/per-device batch: ${GLOBAL_BATCH_SIZE}/${PER_DEVICE_BATCH_SIZE} (gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS})"
echo "effective_global_batch: ${EFFECTIVE_GLOBAL_BATCH}"
echo "target_optimizer_steps: ${STEPS}"
echo "expected_micro_steps: ${EXPECTED_MICRO_STEPS}"
echo "output dir: ${OUTPUT_DIR}"
echo "checkpoint format: PEFT adapter safetensors + full Action Expert modules; resume: ${RESUME}"
printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'

if [[ ${DRY_RUN} == true ]]; then exit 0; fi

[[ -x ${PYTHON_BIN} ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -x ${TRAIN_BIN} ]] || die "training entry point not found: ${TRAIN_BIN}"
${PYTHON_BIN} -c 'import peft' >/dev/null 2>&1 || \
  die "PEFT is not installed in ${PYTHON_BIN}; install the repository's peft extra before saving the image."
for required in \
  "${DATA_ROOT}/meta/info.json" \
  "${BASE_POLICY_ROOT}/config.json" \
  "${BASE_POLICY_ROOT}/model.safetensors" \
  "${BASE_POLICY_ROOT}/policy_preprocessor.json" \
  "${TOKENIZER_ROOT}/tokenizer.model" \
  "${TOKENIZER_ROOT}/tokenizer.json"; do
  [[ -f ${required} ]] || die "missing required offline resource: ${required}"
done
if [[ ${RESUME} == true ]]; then
  [[ -f ${RESUME_CONFIG} ]] || die "resume config not found: ${RESUME_CONFIG}"
elif [[ -e ${OUTPUT_DIR} ]]; then
  if [[ ${OVERWRITE} == true ]]; then rm -rf -- "${OUTPUT_DIR}"; else die "output exists; use --resume or --overwrite: ${OUTPUT_DIR}"; fi
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${DATASET_CACHE}"
cd "${REPO}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export UV_LINK_MODE=copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
