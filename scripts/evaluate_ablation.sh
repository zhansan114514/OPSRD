#!/usr/bin/env bash
set -euo pipefail

# Set CHECKPOINT to "base" for the unadapted model.
CHECKPOINT="${1:-base}"
RUN_TAG="${2:-base}"

WORK_ROOT="${WORK_ROOT:-${PWD}/runs}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PARALLEL_DATASETS="${PARALLEL_DATASETS:-1}"
VAL_N="${VAL_N:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
EVAL_SEED="${EVAL_SEED:-0}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25}"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results/${RUN_TAG}}"

export WORK_ROOT MODEL_NAME_OR_PATH HF_HOME CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${RESULT_ROOT}"

checkpoint_args=()
if [[ "${CHECKPOINT}" != "base" ]]; then
    checkpoint_args=(--checkpoint_dir "${CHECKPOINT}")
fi

sample_args=()
if [[ -n "${NUM_SAMPLES:-}" ]]; then
    sample_args=(--num_samples "${NUM_SAMPLES}")
fi

system_prompt_args=()
if [[ -n "${SYSTEM_PROMPT:-}" ]]; then
    system_prompt_args=(--system_prompt "${SYSTEM_PROMPT}")
fi

problem_prefix_args=()
if [[ -n "${PROBLEM_PREFIX:-}" ]]; then
    problem_prefix_args=(--problem_prefix "${PROBLEM_PREFIX}")
fi

run_dataset() {
    local dataset="$1"
    local visible_gpu="$2"
    CUDA_VISIBLE_DEVICES="${visible_gpu}" python eval/evaluate_math.py \
        --base_model "${MODEL_NAME_OR_PATH}" \
        --dataset "${dataset}" \
        --val_n "${VAL_N}" \
        --temperature 1.0 \
        --top_p 0.95 \
        --top_k -1 \
        --min_p 0.0 \
        --presence_penalty 0.0 \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --seed "${EVAL_SEED}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --gpu_memory_utilization 0.82 \
        --output_file "${RESULT_ROOT}/${dataset}.json" \
        "${sample_args[@]}" \
        "${system_prompt_args[@]}" \
        "${problem_prefix_args[@]}" \
        "${checkpoint_args[@]}"
}

if [[ "${PARALLEL_DATASETS}" == "1" ]]; then
    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    read -r -a dataset_names <<< "${DATASETS}"
    required_gpus=$(( ${#dataset_names[@]} * TENSOR_PARALLEL_SIZE ))
    if (( required_gpus > ${#gpu_ids[@]} )); then
        echo "Parallel evaluation needs ${required_gpus} GPUs: ${#dataset_names[@]} datasets x TP${TENSOR_PARALLEL_SIZE}." >&2
        exit 1
    fi

    pids=()
    for index in "${!dataset_names[@]}"; do
        group_start=$(( index * TENSOR_PARALLEL_SIZE ))
        gpu_group=""
        for (( offset=0; offset<TENSOR_PARALLEL_SIZE; offset++ )); do
            gpu_id="${gpu_ids[$(( group_start + offset ))]}"
            if [[ -n "${gpu_group}" ]]; then
                gpu_group+=","
            fi
            gpu_group+="${gpu_id}"
        done
        run_dataset "${dataset_names[$index]}" "${gpu_group}" &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    exit "${failed}"
else
    for dataset in ${DATASETS}; do
        run_dataset "${dataset}" "${CUDA_VISIBLE_DEVICES}"
    done
fi
