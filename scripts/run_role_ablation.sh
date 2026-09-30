#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_role_ablation.sh solution 1.0 opsd_solution
#   bash scripts/run_role_ablation.sh role     1.0 role_full
#   bash scripts/run_role_ablation.sh role     0.5 role_entropy50
#   bash scripts/run_role_ablation.sh role     0.5 role_guided_entropy50 role
#   bash scripts/run_role_ablation.sh none     0.5 student_only_entropy50 role

TEACHER_CONTEXT_MODE="${1:-solution}"
ENTROPY_TOP_FRACTION="${2:-1.0}"
RUN_TAG="${3:-${TEACHER_CONTEXT_MODE}_entropy${ENTROPY_TOP_FRACTION}}"
STUDENT_CONTEXT_MODE="${4:-none}"

WORK_ROOT="${WORK_ROOT:-${PWD}/runs}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/outputs}"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
USE_VLLM="${USE_VLLM:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12961}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"
LOSS_BETA="${LOSS_BETA:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SEED="${SEED:-42}"

export WORK_ROOT MODEL_NAME_OR_PATH OUTPUT_ROOT HF_HOME CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"

vllm_args=()
if [[ "${USE_VLLM}" == "1" ]]; then
    vllm_args=(
        --use_vllm
        --vllm_mode colocate
        --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    )
    if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
        vllm_args+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
    fi
fi

run_output="${OUTPUT_ROOT}/${RUN_TAG}"
if [[ "${AUTO_RESUME}" == "1" && -d "${run_output}" ]]; then
    latest_checkpoint=""
    while IFS= read -r checkpoint_candidate; do
        if [[ -s "${checkpoint_candidate}/trainer_state.json" \
            && -s "${checkpoint_candidate}/adapter_model.safetensors" ]]; then
            latest_checkpoint="${checkpoint_candidate}"
            break
        fi
    done < <(find "${run_output}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' | sort -Vr)
    if [[ -n "${latest_checkpoint}" ]]; then
        export OPSD_RESUME_FROM_CHECKPOINT="${latest_checkpoint}"
        echo "Auto-resuming ${RUN_TAG} from ${OPSD_RESUME_FROM_CHECKPOINT}"
    fi
fi

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    opsd_train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --output_dir "${OUTPUT_ROOT}" \
    --run_config "${RUN_TAG}" \
    --max_steps "${MAX_STEPS}" \
    --num_train_epochs 30 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 25 \
    --save_total_limit 4 \
    --logging_steps 2 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --beta "${LOSS_BETA}" \
    "${vllm_args[@]}" \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip "${JSD_TOKEN_CLIP}" \
    --teacher_context_mode "${TEACHER_CONTEXT_MODE}" \
    --student_context_mode "${STUDENT_CONTEXT_MODE}" \
    --entropy_top_fraction "${ENTROPY_TOP_FRACTION}" \
    --seed "${SEED}" \
    --report_to none \
    --wandb_project OPSD-role-privileged
