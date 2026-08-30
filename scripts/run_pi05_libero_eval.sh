#!/usr/bin/env bash
set -Eeuo pipefail

VLA_ROOT="${VLA_ROOT:-/path/to/vla}"
REPO="${LEROBOT_REPO:-${VLA_ROOT}/code/lerobot}"
ASSETS_ROOT="${ASSETS_ROOT:-${VLA_ROOT}/assets/libero}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-${VLA_ROOT}/models/paligemma-3b-pt-224}"
RUN_ROOT="${RUN_ROOT:-${VLA_ROOT}/outputs/lerobot/eval}"
LOG_ROOT="${LOG_ROOT:-${VLA_ROOT}/logs/lerobot/eval}"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"

POLICY_PATH=
SUITE=libero_spatial
TASK_IDS=
EPISODES_PER_TASK=50
ENV_BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
SEED=1000
OUTPUT_DIR=
RECORDING=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/run_pi05_libero_eval.sh --policy PATH [options]

  --policy PATH               LeRobot pretrained checkpoint directory.
  --suite NAME                libero_spatial, libero_object, libero_goal, libero_10, or libero_90.
  --task-ids IDS              Comma-separated task ids; default evaluates the whole suite.
  --episodes-per-task N       Rollouts for each selected task (default: 50).
  --env-batch-size N          Vector environments per task (default: 1).
  --max-parallel-tasks N      Tasks evaluated concurrently (default: 1).
  --seed N                    First deterministic rollout seed (default: 1000).
  --output-dir PATH           Default: CFS eval directory with a timestamp.
  --recording                 Save complete rollouts as a local LeRobot dataset.
  --dry-run                   Validate resources and print the command only.
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
    --policy) (($# >= 2)) || die "--policy requires a value"; POLICY_PATH=$2; shift 2 ;;
    --policy=*) POLICY_PATH=${1#*=}; shift ;;
    --suite) (($# >= 2)) || die "--suite requires a value"; SUITE=$2; shift 2 ;;
    --suite=*) SUITE=${1#*=}; shift ;;
    --task-ids) (($# >= 2)) || die "--task-ids requires a value"; TASK_IDS=$2; shift 2 ;;
    --task-ids=*) TASK_IDS=${1#*=}; shift ;;
    --episodes-per-task) (($# >= 2)) || die "--episodes-per-task requires a value"; EPISODES_PER_TASK=$2; shift 2 ;;
    --episodes-per-task=*) EPISODES_PER_TASK=${1#*=}; shift ;;
    --env-batch-size) (($# >= 2)) || die "--env-batch-size requires a value"; ENV_BATCH_SIZE=$2; shift 2 ;;
    --env-batch-size=*) ENV_BATCH_SIZE=${1#*=}; shift ;;
    --max-parallel-tasks) (($# >= 2)) || die "--max-parallel-tasks requires a value"; MAX_PARALLEL_TASKS=$2; shift 2 ;;
    --max-parallel-tasks=*) MAX_PARALLEL_TASKS=${1#*=}; shift ;;
    --seed) (($# >= 2)) || die "--seed requires a value"; SEED=$2; shift 2 ;;
    --seed=*) SEED=${1#*=}; shift ;;
    --output-dir) (($# >= 2)) || die "--output-dir requires a value"; OUTPUT_DIR=$2; shift 2 ;;
    --output-dir=*) OUTPUT_DIR=${1#*=}; shift ;;
    --recording) RECORDING=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n ${POLICY_PATH} ]] || die "--policy is required."
case ${SUITE} in
  libero_spatial|libero_object|libero_goal|libero_10|libero_90) ;;
  *) die "unsupported LIBERO suite: ${SUITE}" ;;
esac
require_positive_integer --episodes-per-task "${EPISODES_PER_TASK}"
require_positive_integer --env-batch-size "${ENV_BATCH_SIZE}"
require_positive_integer --max-parallel-tasks "${MAX_PARALLEL_TASKS}"
require_nonnegative_integer --seed "${SEED}"

TASK_IDS_ARG=
if [[ -n ${TASK_IDS} ]]; then
  compact_ids=${TASK_IDS//[[:space:]]/}
  [[ ${compact_ids} =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--task-ids must be comma-separated non-negative integers."
  TASK_IDS_ARG="[${compact_ids}]"
fi

if [[ -z ${OUTPUT_DIR} ]]; then
  policy_name=$(basename "${POLICY_PATH%/}")
  OUTPUT_DIR=${RUN_ROOT}/${policy_name}_${SUITE}_$(date +%Y%m%d_%H%M%S)
fi
LOG_FILE=${LOG_ROOT}/$(basename "${OUTPUT_DIR}").console.log

CMD=(
  "${PYTHON_BIN}" -m lerobot.scripts.lerobot_eval
  --env.type=libero
  --env.task="${SUITE}"
  --env.observation_height=256
  --env.observation_width=256
  --env.obs_type=pixels_agent_pos
  --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}"
  --eval.n_episodes="${EPISODES_PER_TASK}"
  --eval.batch_size="${ENV_BATCH_SIZE}"
  --eval.use_async_envs=true
  --eval.recording="${RECORDING}"
  --policy.path="${POLICY_PATH}"
  --policy.tokenizer_name="${TOKENIZER_ROOT}"
  --policy.dtype=bfloat16
  --policy.use_amp=true
  '--rename_map={"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
  --output_dir="${OUTPUT_DIR}"
  --seed="${SEED}"
)
if [[ -n ${TASK_IDS_ARG} ]]; then CMD+=(--env.task_ids="${TASK_IDS_ARG}"); fi

echo "policy: ${POLICY_PATH}"
echo "suite/task ids: ${SUITE}/${TASK_IDS:-all}"
echo "episodes per task: ${EPISODES_PER_TASK}"
echo "env batch/max parallel tasks: ${ENV_BATCH_SIZE}/${MAX_PARALLEL_TASKS}"
echo "output: ${OUTPUT_DIR}"
printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'

[[ -x ${PYTHON_BIN} ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -d ${ASSETS_ROOT} ]] || die "LIBERO assets not found: ${ASSETS_ROOT}"
[[ -f ${TOKENIZER_ROOT}/tokenizer.json ]] || die "tokenizer not found: ${TOKENIZER_ROOT}"
[[ -f ${POLICY_PATH}/config.json ]] || die "policy config not found: ${POLICY_PATH}/config.json"
compgen -G "${POLICY_PATH}/*.safetensors" >/dev/null || \
  die "no safetensors weights found directly under: ${POLICY_PATH}"
${PYTHON_BIN} -c 'import libero, mujoco, robosuite' >/dev/null || die "LIBERO runtime import failed."

if [[ ${DRY_RUN} == true ]]; then exit 0; fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
[[ ! -e ${OUTPUT_DIR} ]] || die "output already exists: ${OUTPUT_DIR}"
cd "${REPO}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
echo "eval_exit_code=${status}" | tee -a "${LOG_FILE}"
exit "${status}"
