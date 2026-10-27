#!/bin/bash
#SBATCH --job-name=pf_spec
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# SpecPrefill baseline vs our projected fill, on ONE L40S GPU, no training.
# For each small model (0.6B r2_mix8k projector, 1.7B r5_s17_normB projector; the same small model is SpecPrefill's speculator):
#   ours      attn_small:{5,10,20}%:sent             recompute chosen middle sentences, projected KV for the rest
#   drop      drop-attn_small:{5,10,20}%:sent        identical selection, the rest DROPPED (controlled ablation)
#   spec      spec:{10,20}%                          our method with SpecPrefill's token scores / 32-token chunks
#   baseline  specprefill:{10,20,30,50}% (+10% lah1) SpecPrefill as published (look-ahead 8, pool 13, 32-token chunks,
#                                                    whole prompt, original positions)
#   refs      big, small, hyb
# Tasks: TTFT, NIAH, RULER 4k/8k, QA; GSM8K + code for a subset. Results: results/spec_06/, results/spec_17/
# (summary.md, figures/pareto.png). Restarting skips finished tasks.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
Q=big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.05:sent,drop-attn_small:0.1:sent,drop-attn_small:0.2:sent,spec:0.1,spec:0.2,specprefill:0.1,specprefill:0.2,specprefill:0.3,specprefill:0.5,specprefill:0.1:lah1
SPEED="--speed_configs $Q --speed_lengths 4096,8192,16384,32768"
G=big,hyb,attn_small:0.1:sent,drop-attn_small:0.1:sent,specprefill:0.1,specprefill:0.3
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }

step "smoke test (CPU, tiny models)"
(cd tests && python smoke_specprefill.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert" | tail -5)
for P in "06 Qwen3-0.6B r2_mix8k" "17 Qwen3-1.7B r5_s17_normB"; do
  set -- $P
  step "$2: speed, NIAH, RULER, QA"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/spec_$1 \
    --tasks speed,niah,ruler,qa --configs $Q $SPEED
  step "$2: GSM8K + code"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/spec_$1 \
    --tasks gsm8k,code --configs $G
done
step "all done"
