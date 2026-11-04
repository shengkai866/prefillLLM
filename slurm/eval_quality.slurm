#!/bin/bash
#SBATCH --job-name=pf_quality
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
# QuALITY (dev, 2086 questions on 230 articles of 2.7k-8.4k tokens; scripts/prepare_quality.py) for one trained pair:
#   single question  scripts/eval_recompute.py --tasks quality   -> results/quality_<PAIR>     (letter from first-token logits)
#   multi-question   scripts/eval_multiq.py --pool quality_multiq -> results/quality_mq_<PAIR>  (all questions of an article
#                                                                                                on one document cache)
#   sbatch --export=ALL,PAIR=06 slurm/eval_quality.slurm        Qwen3-0.6B -> Qwen3-8B   (r2_mix8k)        L40S
#   sbatch --export=ALL,PAIR=17 ...                             Qwen3-1.7B -> Qwen3-8B   (r5_s17_normB)    L40S
#   sbatch --export=ALL,PAIR=ll ...                             Llama-3.2-1B -> Llama-3.1-8B (ll1to8_mix8k) L40S
#   sbatch --partition=drjieliu-h100 --export=ALL,PAIR=32 ...   Qwen3-1.7B -> Qwen3-32B  (s17to32n_mix8k)  H100
#   sbatch --export=ALL,PAIR=xq17ll8 | xll1q8 ...               cross-family pairs (after slurm/run_cross.slurm finished)
# Pools are built on first use (CPU, ~1 min) in the large model's data tree. A 4-question smoke run comes first.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
case ${PAIR:?PAIR=06|17|ll|32|xq17ll8|xll1q8} in
  06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt ;;
  17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt ;;
  32) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B; P=$CK/s17to32n_mix8k/projector.pt ;;
  ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt ;;
  xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
           export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/q17toll8_mix8k/projector.pt ;;
  xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
[ -f $P ] || { echo "missing projector $P"; exit 1; }
echo "PAIR=$PAIR small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} projector=$P"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
SMALL=small,
[[ $PAIR == x* ]] && SMALL=""  # cross-family: the small model cannot read the large model's tokens on its own
CFG=big,${SMALL}hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
MQ=big,hyb,ours:0.1:sent,ours-acc:0.03:sent,ours-acc:0.01:sent,ours-each:0.1:sent,spec-agn:0.1,spec-agn:0.3,spec-q1:0.1,spec-each:0.1
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1"; echo "!!!!!!!! FAILED: $*"; }; }

step "pools"
python scripts/prepare_quality.py || exit 1
step "smoke: 4 questions, 2 articles"
python scripts/eval_recompute.py --projector $P --out_dir results/quality_smoke_$PAIR --tasks quality --n_quality 4 --configs big,hyb,attn_small:0.1:sent,specprefill:0.1 --force || exit 1
python scripts/prepare_quality.py --n_articles 2 --suffix _smoke && \
python scripts/eval_multiq.py --projector $P --out_dir results/quality_smoke_$PAIR --pool quality_multiq_smoke --tasks qa --configs big,ours-acc:0.03:sent,spec-each:0.1 --q_max 320 --force || exit 1
step "single question"
ev python scripts/eval_recompute.py --projector $P --out_dir results/quality_$PAIR --tasks quality --configs $CFG
step "multi-question (all questions of an article on one cache)"
ev python scripts/eval_multiq.py --projector $P --out_dir results/quality_mq_$PAIR --pool quality_multiq --tasks qa --configs $MQ --q_max 320  # question blocks are up to 270 tokens
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
