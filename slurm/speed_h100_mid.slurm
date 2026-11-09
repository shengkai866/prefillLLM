#!/bin/bash
#SBATCH --job-name=pf_h100spd
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# TTFT on ONE H100 for the 32B iso-latency comparison (every method CUDA-graphed, fused fold):
#   32      Qwen3-1.7B -> Qwen3-32B: 32B / 1.7B alone / projected / ours 10 % / SpecPrefill 10, 20 %  -> results/h100_32_speed
#   8       Qwen3-8B alone (and 1.7B alone) on the same GPU type                                      -> results/h100_8_speed
#   4       Qwen3-4B alone (loaded as `small` next to 8B, 8B again as reference)                       -> results/h100_4_speed
# The old e32_speed (2026-10-31) used the eager fold and had no 8B / 4B on H100.
#   sbatch slurm/speed_h100_mid.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
LEN=${LEN:-4096,8192,16384}
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $S"; echo "!!!!!!!! FAILED: $*"; }; }
for S in ${STEPS:-32 8 4}; do
  step "$S"
  case $S in
    32) ( export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B
          python scripts/eval_recompute.py --projector $CK/s17to32n_mix8k/projector.pt --out_dir results/h100_32_speed --tasks speed_allgraph \
            --configs hyb --speed_configs big,small,hyb,attn_small:0.1:sent,specprefill:0.1,specprefill:0.2 --speed_lengths $LEN --force ) || FAILED="$FAILED 32" ;;
    8)  ( export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
          python scripts/eval_recompute.py --projector $CK/r5_s17_normB/projector.pt --out_dir results/h100_8_speed --tasks speed_allgraph \
            --configs hyb --speed_configs big,small,hyb,attn_small:0.1:sent,specprefill:0.1 --speed_lengths $LEN --force ) || FAILED="$FAILED 8" ;;
    4)  ( export PREFILL_SMALL_DIR=$M/Qwen3-4B
          python scripts/eval_recompute.py --projector $CK/r2_mix8k/projector.pt --out_dir results/h100_4_speed --tasks speed_allgraph \
            --configs small --speed_configs big,small --speed_lengths $LEN --force ) || FAILED="$FAILED 4" ;;
    *) echo "unknown step $S"; FAILED="$FAILED $S" ;;
  esac
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
