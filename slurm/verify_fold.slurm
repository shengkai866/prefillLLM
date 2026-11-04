#!/bin/bash
#SBATCH --job-name=pf_vfold
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold/logs/%x-%j.out
# Does the fused fold give the same model? Real 8B weights + trained projector (0.6B, r2_mix8k):
#  1. bench_speed --sections check, eager and fused: KL of the fast hybrid vs the reference (HF/training-path) hybrid on
#     held-out text, argmax agreement, graph vs eager logits -> results/verify_fold_check_{eager,fused}_06
#  2. task-level quality with the fused fold (NIAH, RULER 4k/8k, QA, GSM8K; hyb + recompute 10%) -> results/verify_fold_quality_06,
#     to be compared with the eager numbers in results/spec_06 (same examples).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold || exit 1
export TRITON_CACHE_DIR=$PREFILL_ROOT/triton_cache
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-0.6B
P=$PREFILL_ROOT/checkpoints/r2_mix8k/projector.pt
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$PREFILL_FOLD"; echo "!!!!!!!! FAILED: $*"; }; }
for mode in eager fused; do
  export PREFILL_FOLD=$mode
  step "check: fast hybrid vs reference hybrid, fold=$mode"
  ev python scripts/bench_speed.py --projector $P --sections check --out_dir results/verify_fold_check_${mode}_06
done
export PREFILL_FOLD=fused
step "quality with the fused fold"
ev python scripts/eval_recompute.py --projector $P --out_dir results/verify_fold_quality_06 --tasks niah,ruler,qa,gsm8k \
  --configs hyb,attn_small:0.1:sent
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
