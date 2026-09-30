# OPSRD: On-Policy Self-Role Distillation

This repository contains the training, evaluation, analysis, and test code for
**OPSRD**, an improvement to On-Policy Self-Distillation (OPSD). OPSRD uses an
expert role as privileged teacher context, evaluates the student's own prefixes,
and distills the resulting token distribution with forward KL. The student
remains role-free at deployment.

The implementation extends the public OPSD codebase:
[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD). The OPSRD additions are
role-conditioned teacher supervision, high-entropy position selection,
forward/reverse KL and Jensen--Shannon objectives, contribution clipping,
role-placement controls, and compatibility with recent vLLM sampling APIs.

## Repository layout

```text
opsd_trainer.py          # OPSRD loss, teacher context, clipping, and masks
data_collator.py         # Role, neutral, solution, and plain prompt layouts
opsd_train.py            # Training entry point
eval/evaluate_math.py    # vLLM evaluation with prompt controls
scripts/run_role_ablation.sh
                          # Main OPSRD training entry point
scripts/evaluate_ablation.sh
                          # Evaluation entry point
analysis/                # Result summaries and replication gates
tests/                   # Prompt-isolation and objective tests
environment.yml          # Recorded Python package environment
```

The repository contains source code and tests only. Model checkpoints, saved
answers, private server outputs, credentials, and local filesystem paths are not
included.

## Relation to the paper

The paper's main recipe is the `role` teacher with a `0.5` entropy fraction and
a role-free student:

| Setting | Paper value |
| --- | --- |
| Teacher context | Expert role, no reference solution |
| Student context | Plain role-free prompt during training and evaluation |
| Selected positions | Highest-entropy 50% of valid student positions |
| Divergence | Forward KL, `LOSS_BETA=0` |
| Contribution cap | `0.05` for 1.7B and 4B; `0.06` for 8B |
| Optimizer steps | 100 |
| Effective batch size | 36 |
| Training rollout | One per problem, at most 1,024 tokens |
| Training temperature / top-p / top-k | 1.1 / 0.95 / 20 |
| Precision / attention | BF16 / SDPA |
| Evaluation samples | 12 per problem |
| Evaluation temperature / top-p | 1.0 / 0.95 |
| Evaluation generation cap / context limit | 38,912 / 40,960 tokens |

Training uses the 29,434-example train split of
`siyanzhao/Openthoughts_math_30k_opsd`. Evaluation uses AIME 2024, AIME 2025,
and HMMT February 2025, with 30 problems per benchmark.

## Installation

The recorded environment uses Python 3.10, PyTorch 2.8, Transformers 4.57,
TRL 0.26, PEFT 0.17, and vLLM 0.11.

```bash
conda env create -f environment.yml
conda activate opsd
```

Install a CUDA-compatible `flash-attn` build only when your platform requires
it. The paper's primary OPSRD runs use SDPA through
`ATTN_IMPLEMENTATION=sdpa`.

## Training the paper's OPSRD model

Run commands from the repository root. The command below reproduces the
Qwen3-1.7B primary protocol. It explicitly sets every paper-critical value so
that it does not depend on a local machine's defaults:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B \
WORK_ROOT=./runs/opsrd_qwen3_1p7b \
OUTPUT_ROOT=./runs/opsrd_qwen3_1p7b/outputs \
HF_HOME=./runs/opsrd_qwen3_1p7b/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2 \
NUM_PROCESSES=3 \
PER_DEVICE_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=6 \
MAX_STEPS=100 \
MAX_COMPLETION_LENGTH=1024 \
MAX_LENGTH=20000 \
ATTN_IMPLEMENTATION=sdpa \
LOSS_BETA=0 \
JSD_TOKEN_CLIP=0.05 \
USE_VLLM=1 \
bash scripts/run_role_ablation.sh role 0.5 opsrd_qwen3_1p7b_forward_kl none
```

The positional arguments are, in order, `teacher_context`,
`entropy_fraction`, `run_tag`, and `student_context`. The command therefore
means: role-conditioned teacher, highest-entropy half, role-free student, and
forward KL. The fixed teacher is the base model with the student LoRA adapter
disabled. No reference solution is passed to the role teacher.

The reported larger-model runs keep the same effective batch size and paper
protocol while changing the memory layout:

| Model | GPUs | Per-device batch | Gradient accumulation | Clip |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 3 | 2 | 6 | 0.05 |
| Qwen3-4B | 4 | 1 | 9 | 0.05 |
| Qwen3-8B | 2 | 1 | 18 | 0.06 |

Set the corresponding `MODEL_NAME_OR_PATH`, `CUDA_VISIBLE_DEVICES`,
`NUM_PROCESSES`, `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, and
`JSD_TOKEN_CLIP` values before invoking the same `run_role_ablation.sh`
command. The final checkpoint is written under
`OUTPUT_ROOT/<run_tag>/checkpoint-100`.

## Baselines and legacy launchers

The files named `scripts/run_opsd_1b.sh`, `scripts/run_opsd_4b.sh`, and
`scripts/run_opsd_8b.sh` are retained as Answer OPSD baseline launchers. They
use the solution-conditioned teacher, full-position distillation, and the older
FlashAttention-oriented experiment settings. They are useful for baseline
comparisons, but they do not launch the OPSRD method reported in the paper.
Use `scripts/run_role_ablation.sh` for OPSRD.

The `solution` teacher mode in `run_role_ablation.sh` reproduces the
reference-solution OPSD control. The `neutral` and `none` teacher modes support
the controls described in the paper. `LOSS_BETA=0`, `0.5`, and `1` select
forward KL, Jensen--Shannon divergence, and reverse KL, respectively.

## Evaluation with the paper protocol

`scripts/evaluate_ablation.sh` has smaller pilot defaults for quick local
checks. Those defaults are not the paper evaluation. Set the complete paper
protocol explicitly:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B \
WORK_ROOT=./runs/opsrd_qwen3_1p7b \
HF_HOME=./runs/opsrd_qwen3_1p7b/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2 \
DATASETS="aime24 aime25 hmmt25" \
VAL_N=12 \
MAX_NEW_TOKENS=38912 \
MAX_MODEL_LEN=40960 \
EVAL_SEED=0 \
PARALLEL_DATASETS=1 \
TENSOR_PARALLEL_SIZE=1 \
SYSTEM_PROMPT="" \
bash scripts/evaluate_ablation.sh base paper_base_qwen3_1p7b
```

To evaluate the trained OPSRD checkpoint, replace `base` with the checkpoint
path, for example:

```bash
bash scripts/evaluate_ablation.sh \
  ./runs/opsrd_qwen3_1p7b/outputs/opsrd_qwen3_1p7b_forward_kl/checkpoint-100 \
  paper_opsrd_qwen3_1p7b
```

The evaluation prompt enables Qwen3 thinking, uses the ordinary role-free user
prompt, temperature 1.0, top-p 0.95, disabled top-k, and 12 samples per
problem. Set `SYSTEM_PROMPT` only for the direct-prompt or placement controls;
set `PROBLEM_PREFIX="Problem: "` only for the prompt-prefix control.

## Analysis and tests

The prompt-isolation and objective tests can run without model weights:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Result summaries and paired bootstrap gates are in `analysis/`. The full
benchmark evaluation requires GPUs, vLLM, the public datasets, and the Qwen3
checkpoints. Reported paper results were produced with the paper settings above
and the exact prompts described in the accompanying supplementary material.

## Attribution and release scope

The implementation builds on the public OPSD repository linked above. Please
retain that attribution when reusing upstream components. This repository is an
anonymized research-code release; it does not redistribute model weights or the
paper's saved evaluation evidence.
