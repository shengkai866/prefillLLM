#!/bin/bash
#SBATCH --job-name=pf_deep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Deep handoff (prefill/deep.py): 0.6B + mix-8k projector provide the first 36-k layers of Qwen3-8B for the middle tokens,
# the 8B runs its last k layers on all prompt tokens (k = 4/8/12/18/24), no retraining. Compared with 8B alone, the plain
# hybrid and selective recompute (0.6B attention, sentences, 10%) on the same examples as results/recompute*.
# Tasks: k-sweep diagnostic (KL / cosine vs k), NIAH, RULER 4k/8k, QA, TTFT on L40S. Results: results/deep_handoff/.
# Restart skips finished tasks.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python tests/smoke_deep.py
python scripts/eval_deep.py --out_dir results/deep_handoff --tasks sweep,niah,ruler,qa,speed
