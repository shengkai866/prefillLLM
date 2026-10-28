#!/bin/bash
#SBATCH --job-name=pf_mqfix
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
# Redo the `big` config of the 0.6B multi-question QA (results/multiq_06): the first run had NaN first-token logits
# ("!!!!" answers) for 25 / 300 questions from uninitialised value rows (prefill/multiq.py fixed 2026-10-27 while
# pf_follow 27989580 was running; its 1.7B run already used the fix). Other configs are kept. Submitted with
# --dependency=afterany:27989580 so it runs after the main job.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-0.6B python scripts/eval_multiq.py \
  --projector $PREFILL_ROOT/checkpoints/r2_mix8k/projector.pt --out_dir results/multiq_06 --tasks qa --redo big
