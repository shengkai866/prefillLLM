#!/bin/bash
#SBATCH --job-name=pf_cmpspec
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=30:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# pf_cmp + pf_spec in ONE job on ONE L40S GPU, no training (eval_cmp_06_17.slurm, then eval_specprefill.slurm).
# Part 1 (~2.5 h left): 0.6B r2_mix8k vs 1.7B r5_s17_normB projector head-to-head -> results/cmp_06/, results/cmp_17/
# Part 2 (~11 h): SpecPrefill baseline vs our projected fill, both small models -> results/spec_06/, results/spec_17/
# Restarting skips finished tasks. A failure in part 1 does not stop part 2; the job exits non-zero if either failed.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

part_cmp() {
  set -e
  RC=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
  SPEED="--speed_configs big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05 --speed_lengths 4096,8192,16384,32768"
  for P in "06 Qwen3-0.6B r2_mix8k" "17 Qwen3-1.7B r5_s17_normB"; do
    set -- $P
    step "cmp $2: speed, RULER / QA / NIAH"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/cmp_$1 \
      --tasks speed,niah,ruler,qa --configs $RC $SPEED
  done
  for P in "06 Qwen3-0.6B r2_mix8k" "17 Qwen3-1.7B r5_s17_normB"; do
    set -- $P
    step "cmp $2: GSM8K + code"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/cmp_$1 \
      --tasks gsm8k,code --configs hyb,attn_small:0.1:sent
  done
}

part_spec() {
  set -e
  Q=big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.05:sent,drop-attn_small:0.1:sent,drop-attn_small:0.2:sent,spec:0.1,spec:0.2,specprefill:0.1,specprefill:0.2,specprefill:0.3,specprefill:0.5,specprefill:0.1:lah1
  SPEED="--speed_configs $Q --speed_lengths 4096,8192,16384,32768"
  G=big,hyb,attn_small:0.1:sent,drop-attn_small:0.1:sent,specprefill:0.1,specprefill:0.3
  step "spec: smoke test (CPU, tiny models)"
  (cd tests && python smoke_specprefill.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert" | tail -5)
  for P in "06 Qwen3-0.6B r2_mix8k" "17 Qwen3-1.7B r5_s17_normB"; do
    set -- $P
    step "spec $2: speed, NIAH, RULER, QA"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/spec_$1 \
      --tasks speed,niah,ruler,qa --configs $Q $SPEED
    step "spec $2: GSM8K + code"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/spec_$1 \
      --tasks gsm8k,code --configs $G
  done
}

# Run each part as a plain statement: inside "( f ) || ..." bash would silently ignore the set -e in f.
( part_cmp ); [ $? -eq 0 ] || { FAILED="$FAILED cmp"; step "part 1 (cmp) FAILED, continuing with part 2"; }
( part_spec ); [ $? -eq 0 ] || FAILED="$FAILED spec"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
