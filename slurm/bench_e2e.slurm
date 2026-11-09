#!/bin/bash
#SBATCH --job-name=pf_e2e
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Serving costs beyond batch-1 TTFT on ONE L40S (scripts/bench_e2e.py; target Qwen3-8B):
#   end-to-end latency (TTFT + graphed greedy decoding of N tokens), memory (weights / prefill peak / KV cache),
#   batched prefill throughput (B = 1, 2, 4, 8), GPU sanity checks.
# Setups (SETUPS, default all):
#   06  Qwen3-0.6B -> 8B (r2_mix8k): big, 0.6B alone, hyb, ours 10 %, SpecPrefill 10 %   -> results/e2e_06
#   17  Qwen3-1.7B -> 8B (r5_s17_normB): big, 1.7B alone, ours 10 %, SpecPrefill 10 %     -> results/e2e_17
#   4B  Qwen3-4B alone (loaded as `small`, 8B again as the in-run reference)              -> results/e2e_4B
# Then scripts/summarize_e2e.py -> results/e2e_summary.md.
#   sbatch slurm/bench_e2e.slurm
#   sbatch --export=ALL,SETUPS="4B" slurm/bench_e2e.slurm
set -o pipefail
CODE=${CODE:-$SLURM_SUBMIT_DIR}  # checkout with prefill/serving.py and scripts/bench_e2e.py (env.sh cds to the main one)
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd $CODE
export PYTHONPATH=$CODE
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
OUT=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results
echo "code=$CODE out=$OUT"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
for S in ${SETUPS:-06 17 4B}; do
  step "setup $S"
  case $S in
    06) ( export PREFILL_SMALL_DIR=$M/Qwen3-0.6B
          python scripts/bench_e2e.py --projector $CK/r2_mix8k/projector.pt --out_dir $OUT/e2e_06 \
            --configs big,small,hyb,attn_small:0.1:sent,specprefill:0.1 --batch_configs big,small,hyb,attn_small:0.1:sent ) || FAILED="$FAILED 06" ;;
    17) ( export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
          python scripts/bench_e2e.py --projector $CK/r5_s17_normB/projector.pt --out_dir $OUT/e2e_17 \
            --tasks decode,memory,batch --configs big,small,attn_small:0.1:sent,specprefill:0.1 --batch_configs big,small,attn_small:0.1:sent ) || FAILED="$FAILED 17" ;;
    4B) ( export PREFILL_SMALL_DIR=$M/Qwen3-4B
          python scripts/bench_e2e.py --projector $CK/r2_mix8k/projector.pt --out_dir $OUT/e2e_4B \
            --tasks decode,memory,batch --configs big,small --batch_configs big,small ) || FAILED="$FAILED 4B" ;;
    *) echo "unknown setup $S"; FAILED="$FAILED $S" ;;
  esac
done
step "summary"
python scripts/summarize_e2e.py --results $OUT || FAILED="$FAILED summary"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
