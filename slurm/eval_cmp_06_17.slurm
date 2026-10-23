#!/bin/bash
#SBATCH --job-name=pf_cmp
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# 0.6B vs 1.7B projector head-to-head on ONE GPU (L40S or H100, whichever is free; for REPORT_2026-10-23_06B_vs_17B.md).
# No training. Both models get the same evals in the same job, so every number is like for like:
#   0.6B  r2_mix8k projector, 1.7B r5_s17_normB projector:
#     TTFT, RULER / QA / NIAH with the round-5 recompute configs, GSM8K + code (plain and 10% sentences)
# Results: results/cmp_06/, results/cmp_17/. Restarting skips finished tasks.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
RC=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
SPEED="--speed_configs big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05 --speed_lengths 4096,8192,16384,32768"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }

step "0.6B: speed, RULER / QA / NIAH"
PREFILL_SMALL_DIR=$M/Qwen3-0.6B python scripts/eval_recompute.py --projector $CK/r2_mix8k/projector.pt --out_dir results/cmp_06 \
  --tasks speed,niah,ruler,qa --configs $RC $SPEED
step "1.7B: speed, RULER / QA / NIAH"
PREFILL_SMALL_DIR=$M/Qwen3-1.7B python scripts/eval_recompute.py --projector $CK/r5_s17_normB/projector.pt --out_dir results/cmp_17 \
  --tasks speed,niah,ruler,qa --configs $RC $SPEED
step "0.6B: GSM8K + code"
PREFILL_SMALL_DIR=$M/Qwen3-0.6B python scripts/eval_recompute.py --projector $CK/r2_mix8k/projector.pt --out_dir results/cmp_06 \
  --tasks gsm8k,code --configs hyb,attn_small:0.1:sent
step "1.7B: GSM8K + code"
PREFILL_SMALL_DIR=$M/Qwen3-1.7B python scripts/eval_recompute.py --projector $CK/r5_s17_normB/projector.pt --out_dir results/cmp_17 \
  --tasks gsm8k,code --configs hyb,attn_small:0.1:sent
step "all done"
