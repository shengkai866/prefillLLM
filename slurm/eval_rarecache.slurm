#!/bin/bash
#SBATCH --job-name=pf_rare
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# RaReCache baseline (arXiv 2610.11358, reimplemented in prefill/rarecache.py) on ONE L40S GPU, both small models -> 8B.
# Per small model (0.6B with projector r2_mix8k, 1.7B with r5_s17_normB; the projector is only used by the reference
# speed configs):
#   1. fit the ridge maps on our stage-2 training pools (scripts/fit_rarecache.py -> checkpoints/rarecache_<small>/fit.pt)
#   2. rarecache:{0,10,20,30,50}% on speed, NIAH, RULER 4k/8k, QA, GSM8K, code -> results/rare_06, results/rare_17
#      (same examples as results/spec_06|17, so their big / hyb / ours / SpecPrefill rows compare directly);
#      speed also re-measures big, hyb, ours 10% / 20% and SpecPrefill 10% / 30% in the same job.
# Restarting skips a finished fit and finished tasks. A failure for one small model does not stop the other.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
R=rarecache:0,rarecache:0.1,rarecache:0.2,rarecache:0.3,rarecache:0.5
SPEED="--speed_configs big,hyb,attn_small:0.1:sent,attn_small:0.2:sent,specprefill:0.1,specprefill:0.3,$R --speed_lengths 4096,8192,16384,32768"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

step "smoke test (CPU, tiny models)"
(cd tests && python smoke_rarecache.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert" | tail -6)

run_one() {  # $1 tag, $2 small model dir name, $3 projector checkpoint
  set -e
  export PREFILL_SMALL_DIR=$M/$2
  step "$2: fit RaReCache maps"
  python scripts/fit_rarecache.py
  step "$2: speed, NIAH, RULER, QA, GSM8K, code"
  python scripts/eval_recompute.py --projector $CK/$3/projector.pt --rarecache_fit $CK/rarecache_${2#Qwen3-}/fit.pt \
    --out_dir results/rare_$1 --tasks speed,niah,ruler,qa,gsm8k,code --configs $R $SPEED
}

# Run each part as a plain statement: inside "( f ) || ..." bash would silently ignore the set -e in f.
( run_one 06 Qwen3-0.6B r2_mix8k ); [ $? -eq 0 ] || { FAILED="$FAILED 0.6B"; step "0.6B FAILED, continuing with 1.7B"; }
( run_one 17 Qwen3-1.7B r5_s17_normB ); [ $? -eq 0 ] || FAILED="$FAILED 1.7B"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
