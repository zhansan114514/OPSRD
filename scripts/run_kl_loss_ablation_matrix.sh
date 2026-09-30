#!/usr/bin/env bash
set -euo pipefail

# Run the missing loss-direction ablations with the same Role-entropy50
# teacher context, data, and optimization settings as the beta=0 runs.
# beta=0   -> KL(teacher || student), forward KL
# beta=0.5 -> Jensen-Shannon divergence
# beta=1   -> KL(student || teacher), reverse KL

WORK_ROOT="${WORK_ROOT:-${PWD}/runs}"
REPO_ROOT="${REPO_ROOT:-${PWD}}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-opsd}"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs/kl_loss_ablation_matrix}"
GPU_IDS_TEXT="${GPU_IDS_TEXT:-0 1 2}"
MODEL_KEYS_TEXT="${MODEL_KEYS_TEXT:-qwen3_1_7b qwen3_4b qwen3_8b}"
LOSS_SPECS_TEXT="${LOSS_SPECS_TEXT:-jsd:0.5 reverse_kl:1}"
EVAL_SEED="${EVAL_SEED:-0}"
VAL_N="${VAL_N:-12}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}" "${HF_HOME}"
cd "${REPO_ROOT}"
export HF_HOME TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
CONDA_ROOT="${CONDA_BIN%/bin/conda}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/${ENV_NAME}/lib:${CONDA_ROOT}/lib:${LD_LIBRARY_PATH:-}"

read -r -a GPU_IDS <<< "${GPU_IDS_TEXT}"
read -r -a MODEL_KEYS <<< "${MODEL_KEYS_TEXT}"
read -r -a LOSS_SPECS <<< "${LOSS_SPECS_TEXT}"
GPU_COUNT="${#GPU_IDS[@]}"
if (( GPU_COUNT < 2 || GPU_COUNT > 3 )); then
    echo "Two or three GPU IDs are required, got: ${GPU_IDS_TEXT}." >&2
    exit 2
fi
if (( ${#LOSS_SPECS[@]} < 1 )); then
    echo "At least one loss specification is required." >&2
    exit 2
fi
GPU_IDS_CSV="$(IFS=,; echo "${GPU_IDS[*]}")"

exec 9>"${LOG_ROOT}/controller.lock"
if ! flock -n 9; then
    echo "Another KL-loss controller is already running." >&2
    exit 1
fi

run_in_env() {
    "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

set_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" > "${LOG_ROOT}/controller.status"
}

gpu_usage() {
    local gpu="$1"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

wait_for_gpus() {
    local phase="$1"
    local usage_list usage gpu all_free
    while true; do
        usage_list=""
        all_free=1
        for gpu in "${GPU_IDS[@]}"; do
            usage="$(gpu_usage "${gpu}")"
            usage_list+=" gpu${gpu}=${usage}MiB"
            if (( usage > GPU_FREE_THRESHOLD_MIB )); then
                all_free=0
            fi
        done
        if (( all_free == 1 )); then
            return 0
        fi
        set_status "WAITING_GPU phase=${phase}${usage_list}"
        sleep "${GPU_POLL_SECONDS}"
    done
}

model_config() {
    case "$1" in
        qwen3_1_7b)
            MODEL_NAME="Qwen/Qwen3-1.7B"
            PER_DEVICE_BATCH=2
            TRAIN_NUM_PROCESSES="${GPU_COUNT}"
            GRAD_ACCUM=$(( 18 / GPU_COUNT ))
            VLLM_UTIL=0.45
            VLLM_MAX_LEN=""
            VLLM_TP_SIZE=1
            TOKEN_CLIP=0.05
            ;;
        qwen3_4b)
            MODEL_NAME="Qwen/Qwen3-4B"
            PER_DEVICE_BATCH=1
            TRAIN_NUM_PROCESSES="${GPU_COUNT}"
            GRAD_ACCUM=$(( 36 / GPU_COUNT ))
            VLLM_UTIL=0.35
            VLLM_MAX_LEN=""
            VLLM_TP_SIZE=1
            TOKEN_CLIP=0.05
            ;;
        qwen3_8b)
            MODEL_NAME="Qwen/Qwen3-8B"
            PER_DEVICE_BATCH=1
            GRAD_ACCUM=18
            TRAIN_NUM_PROCESSES=2
            VLLM_UTIL=0.30
            VLLM_MAX_LEN=2880
            VLLM_TP_SIZE=2
            TOKEN_CLIP=0.06
            ;;
        *)
            echo "Unknown model key: $1" >&2
            return 2
            ;;
    esac
}

checkpoint_complete() {
    local tag="$1"
    [[ -s "${OUTPUT_ROOT}/${tag}/checkpoint-100/adapter_model.safetensors" \
        && -s "${OUTPUT_ROOT}/${tag}/checkpoint-100/trainer_state.json" ]]
}

results_complete() {
    local tag="$1"
    local paths=()
    local dataset
    for dataset in aime24 aime25 hmmt25; do
        [[ -s "${RESULT_ROOT}/${tag}/${dataset}.json" ]] || return 1
        paths+=("${RESULT_ROOT}/${tag}/${dataset}.json")
    done
    run_in_env python analysis/validate_eval_results.py --expected-val-n "${VAL_N}" "${paths[@]}" \
        >/dev/null 2>&1
}

loss_tag() {
    local model_key="$1"
    local loss_name="$2"
    printf '%s_role_entropy50_%s_seed42%s' "${model_key}" "${loss_name}" "${ARTIFACT_SUFFIX}"
}

train_loss() {
    local model_key="$1"
    local loss_name="$2"
    local beta="$3"
    local tag
    tag="$(loss_tag "${model_key}" "${loss_name}")"
    if checkpoint_complete "${tag}"; then
        set_status "SKIP_TRAIN model=${model_key} loss=${loss_name} tag=${tag} checkpoint_complete"
        return 0
    fi
    wait_for_gpus "train_${model_key}_${loss_name}"
    model_config "${model_key}"
    set_status "TRAIN model=${model_key} loss=${loss_name} beta=${beta} gpus=${GPU_IDS_CSV}"
    MODEL_NAME_OR_PATH="${MODEL_NAME}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HF_HOME="${HF_HOME}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}" \
    NUM_PROCESSES="${TRAIN_NUM_PROCESSES}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH}" \
    GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_UTIL}" \
    USE_VLLM=1 \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_LEN}" \
    VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TP_SIZE}" \
    JSD_TOKEN_CLIP="${TOKEN_CLIP}" \
    LOSS_BETA="${beta}" \
    MAIN_PROCESS_PORT="$((15000 + RANDOM % 1000))" \
    MAX_STEPS=100 \
    SEED=42 \
    AUTO_RESUME=1 \
        run_in_env bash scripts/run_role_ablation.sh role 0.5 "${tag}" none \
        > "${LOG_ROOT}/train_${tag}.log" 2>&1
    checkpoint_complete "${tag}"
}

evaluate_loss() {
    local model_key="$1"
    local loss_name="$2"
    local tag
    tag="$(loss_tag "${model_key}" "${loss_name}")"
    local result_tag="official_${tag}_eval_seed${EVAL_SEED}"
    if results_complete "${result_tag}"; then
        set_status "SKIP_EVAL model=${model_key} loss=${loss_name} tag=${result_tag} results_complete"
        return 0
    fi
    wait_for_gpus "eval_${model_key}_${loss_name}"
    model_config "${model_key}"
    set_status "EVAL model=${model_key} loss=${loss_name} gpus=${GPU_IDS_CSV}"
    local parallel_datasets=0
    if (( GPU_COUNT >= 3 )); then
        parallel_datasets=1
    fi
    MODEL_NAME_OR_PATH="${MODEL_NAME}" \
    WORK_ROOT="${WORK_ROOT}" \
    HF_HOME="${HF_HOME}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS_CSV}" \
    RESULT_ROOT="${RESULT_ROOT}/${result_tag}" \
    VAL_N="${VAL_N}" \
    MAX_NEW_TOKENS=38912 \
    MAX_MODEL_LEN=40960 \
    EVAL_SEED="${EVAL_SEED}" \
    PARALLEL_DATASETS="${parallel_datasets}" \
    TENSOR_PARALLEL_SIZE=1 \
    SYSTEM_PROMPT="" \
        run_in_env bash scripts/evaluate_ablation.sh \
            "${OUTPUT_ROOT}/${tag}/checkpoint-100" "${result_tag}" \
            > "${LOG_ROOT}/eval_${tag}.log" 2>&1
    results_complete "${result_tag}"
}

write_model_summary() {
    local model_key="$1"
    local summary_args=()
    local forward_dir=""
    if [[ "${ARTIFACT_SUFFIX}" == "" ]]; then
        case "${model_key}" in
            qwen3_1_7b) forward_dir="${RESULT_ROOT}/official_role_entropy50_seed42" ;;
            *) forward_dir="${RESULT_ROOT}/official_${model_key}_role_entropy50_seed42" ;;
        esac
    fi
    if [[ -n "${forward_dir}" \
        && -s "${forward_dir}/aime24.json" \
        && -s "${forward_dir}/aime25.json" \
        && -s "${forward_dir}/hmmt25.json" ]]; then
        summary_args+=(--run "forward_kl=${forward_dir}")
    fi
    local loss_name tag result_tag
    for loss_name in jsd reverse_kl; do
        tag="$(loss_tag "${model_key}" "${loss_name}")"
        result_tag="official_${tag}_eval_seed${EVAL_SEED}"
        if results_complete "${result_tag}"; then
            summary_args+=(--run "${loss_name}=${RESULT_ROOT}/${result_tag}")
        fi
    done
    if (( ${#summary_args[@]} < 2 )); then
        return 0
    fi
    run_in_env python analysis/summarize_results.py \
        "${summary_args[@]}" \
        --output-json "${RESULT_ROOT}/kl_ablation_${model_key}${ARTIFACT_SUFFIX}_summary.json" \
        > "${LOG_ROOT}/summary_${model_key}.log" 2>&1
}

trap 'set_status FAILED line=${LINENO}' ERR
set_status STARTING models="${MODEL_KEYS_TEXT}" losses="${LOSS_SPECS_TEXT}" gpus="${GPU_IDS_CSV}"

for model_key in "${MODEL_KEYS[@]}"; do
    for spec in "${LOSS_SPECS[@]}"; do
        loss_name="${spec%%:*}"
        beta="${spec#*:}"
        case "${loss_name}" in
            jsd|reverse_kl) ;;
            *) echo "Unsupported loss name '${loss_name}'; use jsd or reverse_kl." >&2; exit 2 ;;
        esac
        train_loss "${model_key}" "${loss_name}" "${beta}"
        evaluate_loss "${model_key}" "${loss_name}"
    done
    write_model_summary "${model_key}"
done

set_status COMPLETE
