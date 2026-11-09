#!/bin/bash
#SBATCH --job-name=pf_attn_verify
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Quality check of the position-causal attention kernel after the merge (PREFILL_ATTN=triton, the default): unit test,
# then NIAH / RULER / QA for big, hyb, ours 10 %, SpecPrefill 10 % under the kernel, compared with the same configs in
# results/fused_<tag>_main (masked SDPA) -> results/attn_verify_<tag>, results/attn_verify_summary.md.
# Pairs: 06 (main pair), ll1toq8 (the one attn_check cell with top-1 agreement 0.75), ll1to8 (Llama target).
#   sbatch slurm/verify_attn.slurm          PAIRS="06" to restrict
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_ATTN=triton
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"06 xll1q8 ll"}
CFG=big,hyb,attn_small:0.1:sent,specprefill:0.1
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
step "unit test"
python tests/test_attn_fused.py --lengths 4096,8192 --reps 2 || { echo "unit test failed"; exit 1; }
rm -f results/attn_verify_summary.md
for p in $PAIRS; do
  ( unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
    case $p in
      06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt; TAG=06 ;;
      ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt; TAG=ll1to8 ;;
      xll1q8) export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt; TAG=ll1toq8 ;;
      *) echo "unknown pair $p"; exit 1 ;;
    esac
    step "$p: niah, ruler, qa under the kernel"
    python scripts/eval_recompute.py --projector $P --out_dir results/attn_verify_$TAG --tasks niah,ruler,qa --configs $CFG --force && \
    python scripts/compare_main.py results/fused_${TAG}_main results/attn_verify_$TAG --out results/attn_verify_summary.md ) \
  || { FAILED="$FAILED $p"; echo "!!!!!!!! FAILED: $p"; }
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
