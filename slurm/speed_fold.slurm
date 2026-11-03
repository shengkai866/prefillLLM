#!/bin/bash
#SBATCH --job-name=pf_spdfold
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold/logs/%x-%j.out
# TTFT of the whole pipeline with the eager fold (FoldedProjectorKV, = REPORT_2026-10-28 §3.1) vs the fused Triton
# fold (FusedFoldKV), every method CUDA-graphed, SpecPrefill 10/20/30% in the same run; then the per-stage breakdown
# (small prefill / fold / tail) of bench_speed.py in both modes. PAIR=06 (0.6B, r2_mix8k) | 17 (1.7B, r5_s17_normB) -> 8B.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold || exit 1
export TRITON_CACHE_DIR=$PREFILL_ROOT/triton_cache
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
case ${PAIR:?PAIR=06|17} in
  06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt ;;
  17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$PREFILL_FOLD"; echo "!!!!!!!! FAILED: $*"; }; }
GCFG=big,small,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
LEN=4096,8192,16384,32768
for mode in eager fused; do
  export PREFILL_FOLD=$mode
  step "speed_allgraph, fold=$mode"
  ev python scripts/eval_recompute.py --projector $P --out_dir results/speed_fold_${mode}_$PAIR \
    --tasks speed_allgraph --configs hyb --speed_configs $GCFG --speed_lengths $LEN
done
for mode in eager fused; do
  export PREFILL_FOLD=$mode
  step "per-stage breakdown (bench_speed), fold=$mode"
  ev python scripts/bench_speed.py --projector $P --sections speed --lengths $LEN --out_dir results/speed_fold_stages_${mode}_$PAIR
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
