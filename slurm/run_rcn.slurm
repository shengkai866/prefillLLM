#!/bin/bash
#SBATCH --job-name=pf_rcn
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch moe-router: recompute-aware stage 2 with IMPERFECT training selection (rcn). Same recipe as run_rc.slurm (from
# the base projector, 4000 steps, fractions 0/0.1/0.1/0.2) except the selection per sequence is
#   exact 50% (= inference) | miss 25% (top sentences making up half the budget excluded) | random 25%
# so the projector does not count on the most attended sentences always being recomputed (REPORT_2026-11-07 section 3).
# Pairs: ll (Llama 1B->8B), xll1q8 (Llama 1B->Qwen 8B), 17 (Qwen 1.7B->8B). Steps per pair:
#   train -> checkpoints/rcn_<tag>
#   kl    -> results/rcn_kl_<tag>   (base vs rc vs rcn, sd_mixtest, recompute 0 / 0.1)
#   qa    -> results/rc_qa_rcn_<tag> (scripts/diag_rc_qa.py base vs rcn on the QA benchmark prompts)
#   main  -> results/rcn_<tag>_main (niah / ruler / qa / gsm8k / code, hyb and ours 10%)
# Resume: --export=ALL,START_PAIR=<pair>[,START_STEP=<step>]
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"ll xll1q8 17"}
START_PAIR=${START_PAIR:-}
START_STEP=${START_STEP:-}
STEPS=${STEPS:-4000}
RCF=${RCF:-0,0.1,0.1,0.2}
SEL=${SEL:-exact:0.5,miss:0.25,random:0.25}
FT="--max_steps $STEPS --accum 4 --lr 3e-5 --warmup 100 --curriculum_frac 0 --val_every 500 --val_per_type 2 --save_every 250 \
    --save_step_every 2000 --time_budget_hours 9 --sd sd_mix4k,sd_mixlong --max_prompt 8064"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

setup_pair() {
  PAIR=$1
  unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
  EXTRA=""
  case $PAIR in
    06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt; TAG=06 ;;
    17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt; TAG=17; EXTRA="--lambda_attn 1.0 --attn_layers 4" ;;
    ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt; TAG=ll1to8 ;;
    xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/llama_env.sh; export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
             P=$CK/q17toll8_mix8k/projector.pt; TAG=q17toll8 ;;
    xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt; TAG=ll1toq8 ;;
    *) echo "unknown PAIR $PAIR"; return 1 ;;
  esac
  RC=$CK/rc_$TAG
  RCN=$CK/rcn_$TAG
  echo "PAIR=$PAIR small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} data=${PREFILL_DATA_DIR:-default} base=$P"
}

train() {  # $1 out dir, rest: extra args
  local out=$1; shift
  [ -f $out/DONE ] && { echo "$out done, skipped"; return 0; }
  python scripts/train_stage2.py --init $P --out_dir $out $FT $EXTRA "$@" && touch $out/DONE
}

run_step() {
  ( setup_pair $1 || exit 1
    STEP=$2
    step "$PAIR / $STEP"
    case $STEP in
      train) train $RCN --rc_fracs $RCF --rc_sel $SEL ;;
      kl)    python scripts/eval_rc_kl.py --projectors base=$P,rc=$RC/projector.pt,rcn=$RCN/projector.pt --fracs 0,0.1 \
               --out_dir results/rcn_kl_$TAG ;;
      qa)    python scripts/diag_rc_qa.py --base $P --rc $RCN/projector.pt --out_dir results/rc_qa_rcn_$TAG ;;
      main)  python scripts/eval_recompute.py --projector $RCN/projector.pt --out_dir results/rcn_${TAG}_main \
               --tasks niah,ruler,qa,gsm8k,code --configs hyb,attn_small:0.1:sent ;;
    esac )
  local rc=$?
  [ $rc -ne 0 ] && { FAILED="$FAILED $1:$2"; echo "!!!!!!!! FAILED: $1 / $2 (exit $rc)"; }
  return $rc
}

started=0
[ -z "$START_PAIR" ] && started=1
for pair in $PAIRS; do
  [[ $started == 0 && $pair == $START_PAIR ]] && started=1
  [[ $started == 0 ]] && continue
  sstarted=1
  [[ $pair == $START_PAIR && -n $START_STEP ]] && sstarted=0
  for s in train kl qa main; do
    [[ $sstarted == 0 && $s == $START_STEP ]] && sstarted=1
    [[ $sstarted == 0 ]] && continue
    run_step $pair $s
    # a failed training makes the later steps of this pair meaningless
    rcs=$?
    if [[ $rcs -ne 0 && $s == train ]]; then echo "skipping the rest of $pair"; break; fi
  done
done
step "finished; failed steps:${FAILED:- none}"
