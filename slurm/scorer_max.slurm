#!/bin/bash
#SBATCH --job-name=pf_scorer_max
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-scorer/logs/%x-%j.out
# Does SpecPrefill's score AGGREGATION, without its look-ahead, improve our token selection? (scorer attn_max,
# prefill/recompute.py: the same 64 tail queries in the same small-model pass as attn_small, but smoothed along the
# prompt, maxed over heads and layers, averaged over queries). One L40S, runs from the scorer-max worktree; main and the
# other worktrees are untouched. Per pair (results/scorer_<tag>):
#   attn_small:0.1:sent   ours today (mean over heads / queries / layers, sentence units)
#   attn_max:0.1:sent     SpecPrefill aggregation, sentence units
#   attn_max:0.1          SpecPrefill aggregation + SpecPrefill's 32-token chunks by mean (= SpecPrefill scorer minus look-ahead)
#   spec:0.1              full SpecPrefill scorer with 8-step look-ahead (upper reference, slower)
#   attn_small:0.05:sent / attn_max:0.05:sent   smaller budget, where the choice matters more
# tasks: RULER 4k / 8k, QA 4k / 8k, QA 16k / 32k, then TTFT (CUDA graph) of both scorers vs SpecPrefill 10 %.
# -> scripts/summarize_scorer.py -> results/scorer_max_summary.md
#   sbatch slurm/scorer_max.slurm        PAIRS="06" to restrict
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-scorer || exit 1
export PREFILL_FOLD=fused PREFILL_ATTN=triton
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"06 17"}
CFG=attn_small:0.1:sent,attn_max:0.1:sent,attn_max:0.1,spec:0.1,attn_small:0.05:sent,attn_max:0.05:sent
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
git log --oneline -1
python tests/test_attn_max.py || { echo "!!!!!!!! test_attn_max FAILED"; exit 1; }

for p in $PAIRS; do
  ( unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
    case $p in
      06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt; TAG=06 ;;
      17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt; TAG=17 ;;
      *) echo "unknown pair $p"; exit 1 ;;
    esac
    R=results/scorer_$TAG
    step "$p / quality"
    python scripts/eval_recompute.py --projector $P --out_dir $R --tasks ruler,qa,qa_long --configs $CFG --force || exit 1
    step "$p / speed"
    python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph --configs attn_small:0.1:sent \
      --speed_configs big,attn_small:0.1:sent,attn_max:0.1:sent,specprefill:0.1 --speed_lengths 4096,8192,16384,32768 --force || exit 1
  ) || FAILED="$FAILED $p"
done
python scripts/summarize_scorer.py
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
