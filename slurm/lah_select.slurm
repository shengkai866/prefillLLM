#!/bin/bash
#SBATCH --job-name=pf_lah
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Draft-answer selection scorers on ONE L40S (prefill/recompute.py small_prefill_scored_lah; idea from the pseudo-labels
# of Prefilling-dLLM and SpecPrefill's look-ahead): which sentences get recomputed is scored by the attention of the
# question (last 64 prompt tokens, as now) PLUS that of a greedy draft answer of the small model.
#   attn_small:0.1:sent     current scorer (re-run here for paired comparisons)
#   attn_lah:0.1:sent       question + 7 draft tokens        attn_lah32:0.1:sent   question + 31 draft tokens
#   attn_lahonly:0.1:sent   last prompt token + 7 draft tokens only (SpecPrefill's query set, our attention mass)
#   oracle                  recompute the answer positions (main / long only: tasks with answer strings)
# Pairs (PAIRS, default "06 17"): 06 Qwen3-0.6B -> 8B (r2_mix8k), 17 Qwen3-1.7B -> 8B (r5_s17_normB), ll Llama-1B -> 8B.
# Steps (STEPS): speed (TTFT 4k-32k, all graphed), main (RULER, QA), long (16k / 32k QA), quality (QuALITY single question),
#   bench (LastingBench, MuSiQue, Qasper, LongHealth, ContractNLI) -> results/lah_<pair>_<step>; then
#   scripts/summarize_lah.py -> results/lah_summary.md.
#   sbatch slurm/lah_select.slurm
set -o pipefail
CODE=${CODE:-$SLURM_SUBMIT_DIR}
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd $CODE
export PYTHONPATH=$CODE
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
OUT=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results
SEL=attn_small:0.1:sent,attn_lah:0.1:sent,attn_lah32:0.1:sent,attn_lahonly:0.1:sent
echo "code=$CODE out=$OUT"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

run_pair() {
  ( P=$1
    case $P in
      06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; PJ=$CK/r2_mix8k/projector.pt ;;
      17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; PJ=$CK/r5_s17_normB/projector.pt ;;
      ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; PJ=$CK/ll1to8_mix8k/projector.pt ;;
      *) echo "FAILED_PAIR: $P"; exit 1 ;;
    esac
    R=$OUT/lah_$P
    F=""
    ev() { "$@" || { F="$F $P:$S"; echo "!!!!!!!! FAILED: $*"; }; }
    for S in ${STEPS:-speed main long quality bench}; do
      step "$P / $S"
      case $S in
        speed)   ev python scripts/eval_recompute.py --projector $PJ --out_dir ${R}_speed --tasks speed_allgraph --configs big \
                   --speed_configs big,$SEL --speed_lengths 4096,8192,16384,32768 --force ;;
        main)    ev python scripts/eval_recompute.py --projector $PJ --out_dir ${R}_main --tasks ruler,qa --configs $SEL,oracle --force ;;
        long)    ev python scripts/eval_recompute.py --projector $PJ --out_dir ${R}_long --tasks qa_long --configs $SEL,oracle --force ;;
        quality) ev python scripts/eval_recompute.py --projector $PJ --out_dir ${R}_quality --tasks quality --configs $SEL --force ;;
        bench)   ev python scripts/eval_bench.py --projector $PJ --out_dir ${R}_bench --max_len 65536 \
                   --tasks lastingbench,musique,qasper,longhealth,contractnli --configs $SEL --force ;;
        *) echo "unknown step $S"; F="$F $P:$S" ;;
      esac
    done
    [ -z "$F" ] || echo "FAILED_PAIR:$F" )
}

for P in ${PAIRS:-06 17}; do
  out=$(run_pair $P 2>&1 | tee >(cat >&2) | grep "^FAILED_PAIR:" | tail -1)
  [ -z "$out" ] || FAILED="$FAILED ${out#FAILED_PAIR:}"
done
step "summary"
python scripts/summarize_lah.py --results $OUT --pairs "$(echo ${PAIRS:-06 17} | tr ' ' ',')" || FAILED="$FAILED summary"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
