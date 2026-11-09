#!/bin/bash
#SBATCH --job-name=pf_diag
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Diagnostics of where ours loses to the large model on the long-context bench, one L40S, one step after another
# (scripts/diag_bench.py -> results/diag_summary.md):
#   nli   ContractNLI with a floor on recomputed tokens (0.1m256 / 0.1m512 = 10 % but at least 256 / 512) and 20 / 30 %
#   hop   LastingBench + MuSiQue at 20 / 30 % recompute
#   long  LooGLE + LooGLE-long: projection only (hyb), selection only (drop-attn_small:0.1), 20 % — Llama-3.1-8B target
#         (ll) vs Qwen3-8B target (06)
# nli + hop run for SHORT_PAIRS (default all five L40S pairs, ~20 min each), long for LONG_PAIRS (default "ll 06", ~3.5 h
# each). Baselines are reused from results/bench_<pair>. Finished tasks are skipped (rerun with FORCE=1).
#   sbatch slurm/diag_bench.slurm
#   sbatch --export=ALL,SHORT_PAIRS=,LONG_PAIRS=xq17ll8 slurm/diag_bench.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
SHORT_PAIRS=${SHORT_PAIRS-"06 17 ll xq17ll8 xll1q8"}
LONG_PAIRS=${LONG_PAIRS-"ll 06"}
NLI=attn_small:0.1m256:sent,attn_small:0.1m512:sent,attn_small:0.2:sent,attn_small:0.3:sent
HOP=attn_small:0.2:sent,attn_small:0.3:sent
LONG=hyb,drop-attn_small:0.1:sent,attn_small:0.2:sent
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

run() {  # $1 pair, $2 kind, $3 tasks, $4 configs; in a subshell so the pair's environment does not leak
  ( unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
    case $1 in
      06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt ;;
      17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt ;;
      ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt ;;
      xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
               export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/q17toll8_mix8k/projector.pt ;;
      xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt ;;
      *) echo "unknown pair $1"; exit 1 ;;
    esac
    step "$1 / $2 ($3)"
    echo "small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} data=${PREFILL_DATA_DIR:-default} projector=$P fold=$PREFILL_FOLD"
    python scripts/eval_bench.py --projector $P --out_dir results/diag_$2_$1 --max_len 65536 --tasks $3 --configs $4 ${FORCE:+--force} ) \
  || { FAILED="$FAILED $1:$2"; echo "!!!!!!!! FAILED: $1 $2"; }
}

for p in $SHORT_PAIRS; do
  run $p nli contractnli $NLI
  run $p hop lastingbench,musique $HOP
done
python scripts/diag_bench.py
for p in $LONG_PAIRS; do
  run $p long loogle,loogle_long $LONG
  python scripts/diag_bench.py
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
