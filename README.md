# OPSRD: On-Policy Self-Role Distillation

This repository contains the training, evaluation, analysis, and test code for
**OPSRD**, an improvement to On-Policy Self-Distillation (OPSD). OPSRD uses an
expert role as privileged teacher context, scores the student's own prefixes,
and distills the resulting token distribution with a selectable divergence and
entropy-based position selection. The student remains role-free at deployment.

The implementation extends the public OPSD codebase:
[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD). The changes in this
repository add role-conditioned teacher supervision, optional student-role and
neutral controls, highest-entropy position selection, forward/reverse KL and
Jensen–Shannon objectives, contribution clipping, and compatibility with recent
vLLM sampling APIs.

## Repository layout

```text
opsd_trainer.py          # OPSRD loss, teacher context, clipping, and masks
data_collator.py         # Role, neutral, solution, and plain prompt layouts
opsd_train.py            # Training entry point
eval/evaluate_math.py    # vLLM evaluation with prompt controls
scripts/                 # Reproducible training and evaluation launchers
analysis/                # Result summaries and replication gates
tests/                   # Prompt-isolation and objective tests
environment.yml          # Recorded Python package environment
```

The repository contains source code and tests only. Model checkpoints, private
server outputs, credentials, and local filesystem paths are not included.

## Installation

The recorded environment uses Python 3.10, PyTorch 2.8, Transformers 4.57,
TRL 0.26, PEFT 0.17, and vLLM 0.11.

```bash
conda env create -f environment.yml
conda activate opsd
```

Install a CUDA-compatible `flash-attn` build if your platform requires it. The
launchers can also use SDPA by setting `ATTN_IMPLEMENTATION=sdpa`.

## Data and models

Training expects the public `siyanzhao/Openthoughts_math_30k_opsd` dataset. The
main experiments use the public Qwen3 checkpoints `Qwen/Qwen3-1.7B`,
`Qwen/Qwen3-4B`, and `Qwen/Qwen3-8B`. The dataset and model weights are
downloaded by the Hugging Face libraries at run time; they are not redistributed
in this repository.

## Training

Run commands from the repository root. Set `CUDA_VISIBLE_DEVICES`,
`NUM_PROCESSES`, and `MODEL_NAME_OR_PATH` for the available hardware.

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B \
NUM_PROCESSES=3 CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_role_ablation.sh role 0.5 role_entropy50
```

The fourth argument controls the student context. The default `none` keeps the
student role-free; use `role` or `neutral` for placement controls. The first
argument selects the teacher context: `role` is OPSRD, `solution` reproduces
the original OPSD reference-solution teacher, and `none` is a plain teacher.
`LOSS_BETA=0`, `0.5`, and `1` select forward KL, JSD, and reverse KL,
respectively.

For the principal scale settings, the convenience launchers are:

```bash
bash scripts/run_opsd_1b.sh
bash scripts/run_opsd_4b.sh
bash scripts/run_opsd_8b.sh
```

These scripts use public Hugging Face model identifiers by default. Override
`OUTPUT_DIR` and `MODEL_NAME_OR_PATH` when using local checkpoints.

## Evaluation

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B \
DATASETS="aime24 aime25 hmmt25" VAL_N=12 \
bash scripts/evaluate_ablation.sh base base
```

Set `SYSTEM_PROMPT` to evaluate direct role prompting and `PROBLEM_PREFIX` to
reproduce prompt-placement controls. The evaluation script records the exact
prompt template, seed, checkpoint, and sampling parameters in its JSON output.

## Tests

The prompt-isolation and objective tests do not require model weights:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The full benchmark evaluation requires GPUs, vLLM, the public datasets, and the
Qwen3 checkpoints. Reported paper results were produced with the configurations
described in the accompanying supplementary material.

## Reproducibility and attribution

The paper's supplementary release contains the complete prompts, exact
hyperparameters, saved evaluation evidence, and detailed validation records.
This repository is an anonymized research-code release. Please retain the
upstream OPSD attribution when reusing components from the original codebase.
