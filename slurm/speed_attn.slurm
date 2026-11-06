#!/bin/bash
#SBATCH --job-name=pf_attn_speed
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-attn/logs/%x-%j.out
# TTFT with the position-causal attention kernel (prefill/attn_fused.py, PREFILL_ATTN=triton) on top of the fused fold,
# on ONE L40S, all five pairs that fit a 48 GB L40S (the 32B pair needs the H100 and is not here). Runs from the
# attn-kernel worktree; the main checkout and its jobs are untouched. Per pair (results/attn_<tag>_<step>):
#   speed   PREFILL_ATTN=triton: speed_allgraph of large alone / small alone / projected cache / ours 10 % and 20 % /
#           SpecPrefill 10 / 20 / 30 %, + breakdown (per-stage times, incl. the recompute call) + attn_check (real
#           prompts, first-token logits sdpa vs triton)
#   sdpa    PREFILL_ATTN=sdpa (masked SDPA, the current main): projected cache + ours, large alone as the in-run
#           reference, + breakdown
#   qa      (06 only) QA with ours 10 % under the kernel, to compare with results/fused_06_main
# -> scripts/summarize_attn.py -> results/attn_speed_summary.md
#   sbatch slurm/speed_attn.slurm      PAIRS="06 17" STEPS="speed sdpa" to restrict
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-attn || exit 1
export PREFILL_FOLD=fused
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"06 17 ll xq17ll8 xll1q8"}
STEPS=${STEPS:-"speed sdpa qa"}
LEN=4096,8192,16384,32768
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $PAIR:$STEP"; echo "!!!!!!!! FAILED: $*"; }; }
python -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"
git log --oneline -1

setup_pair() {
  PAIR=$1
  unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
  SMALL=small,
  case $PAIR in
    06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt; TAG=06 ;;
    17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt; TAG=17 ;;
    ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt; TAG=ll1to8 ;;
    xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
             P=$CK/q17toll8_mix8k/projector.pt; TAG=q17toll8; SMALL="" ;;
    xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt; TAG=ll1toq8; SMALL="" ;;
    *) echo "unknown PAIR $PAIR"; return 1 ;;
  esac
  [ -f $P ] || { echo "missing projector $P"; return 1; }
  R=results/attn_$TAG
  echo "PAIR=$PAIR small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} projector=$P fold=$PREFILL_FOLD"
}

for p in $PAIRS; do
  for STEP in $STEPS; do
    ( setup_pair $p || exit 1
      step "$PAIR / $STEP"
      case $STEP in
        speed) export PREFILL_ATTN=triton
               ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph,breakdown,attn_check \
                 --configs hyb,attn_small:0.1:sent --n_attn_check 4 --speed_lengths $LEN --force \
                 --speed_configs big,${SMALL}hyb,attn_small:0.1:sent,attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3 ;;
        sdpa)  export PREFILL_ATTN=sdpa
               ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed_sdpa --tasks speed_allgraph,breakdown --configs hyb \
                 --speed_configs big,hyb,attn_small:0.1:sent,attn_small:0.2:sent --speed_lengths $LEN --force ;;
        qa)    [ $PAIR == 06 ] || exit 0
               export PREFILL_ATTN=triton
               ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_qa --tasks qa --configs attn_small:0.1:sent --force ;;
      esac
      [ -z "$FAILED" ] || exit 1 ) || FAILED="$FAILED $p:$STEP"
  done
done
python scripts/summarize_attn.py
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
