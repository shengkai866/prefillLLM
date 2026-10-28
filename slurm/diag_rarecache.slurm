#!/bin/bash
#SBATCH --job-name=pf_rdiag
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Why is our RaReCache reimplementation so weak (results/rare_06: rho 0 RULER 0.215 vs our hyb-no-recompute 0.736)?
# 0.6B -> 8B only, one L40S, ~1-1.5 h:
#   1. scripts/diag_rarecache.py: held-out R^2 of the saved ridge maps, bf16 inference path vs fp32, by position
#      (sink row 0 / 1-7 / rest), and a fresh fp64 fit to separate "W quantised to bf16" from "map is just weak"
#      -> results/rare_diag_0.6B/fit_diag.{json,md}
#   2. quality ablation on QA (30 per source and length) + GSM8K 8-shot (60), same examples as results/rare_06:
#        rarecache:0 / 0.1          as in the paper run (only the last token / top-rho forced)
#        rarecache-s:*              + our sink 8 / tail 64 recomputed by the 8B (is the design handicap the gap?)
#        rarecache-f:*              ridge map applied in fp32 (is bf16 rounding the gap?)
#        rarecache-sf:*             both
#        hyb, attn_small:0.1:sent   our reference rows on the same examples
#      -> results/rare_diag_0.6B (eval_recompute summary.md)
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
export PREFILL_SMALL_DIR=$M/Qwen3-0.6B
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

step "1. map diagnostics (held-out R^2, bf16 vs fp32, by position, refit)"
python scripts/diag_rarecache.py || FAILED="$FAILED diag"

step "2. quality ablation: sink/tail and fp32 variants on QA + GSM8K"
python scripts/eval_recompute.py --projector $CK/r2_mix8k/projector.pt --rarecache_fit $CK/rarecache_0.6B/fit.pt \
  --out_dir results/rare_diag_06 --tasks qa,gsm8k --n_qa 30 --n_gsm8k 60 \
  --configs hyb,attn_small:0.1:sent,rarecache:0,rarecache-s:0,rarecache-f:0,rarecache-sf:0,rarecache:0.1,rarecache-s:0.1,rarecache-sf:0.1 \
  || FAILED="$FAILED eval"

step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
