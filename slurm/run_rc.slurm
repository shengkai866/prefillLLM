#!/bin/bash
#SBATCH --job-name=pf_rc
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch moe-router: recompute-aware stage 2 (prefill/rc_train.py) for every pair that trains on a 48 GB L40S, one after
# the other on ONE L40S. Each pair fine-tunes its final projector for STEPS steps with the large model recomputing a
# fraction (0 / 0.1 / 0.1 / 0.2, drawn per sequence) of the middle tokens chosen like inference ("ours": small-model
# attention, sentences) INSIDE the training forward; the projector only learns the cells that are not recomputed.
#   06      Qwen3-0.6B -> Qwen3-8B         r2_mix8k        -> checkpoints/rc_06       (+ control rcctl_06: same fine-tune, no recompute)
#   17      Qwen3-1.7B -> Qwen3-8B         r5_s17_normB    -> checkpoints/rc_17       (lambda_attn 1.0 as its recipe)
#   ll      Llama-3.2-1B -> Llama-3.1-8B   ll1to8_mix8k    -> checkpoints/rc_ll1to8
#   xll1q8  Llama-3.2-1B -> Qwen3-8B       ll1toq8_mix8k   -> checkpoints/rc_ll1toq8
#   xq17ll8 Qwen3-1.7B -> Llama-3.1-8B     q17toll8_mix8k  -> checkpoints/rc_q17toll8
# Steps per pair: test (tests/test_rc_train.py), train, [ctl (06 only)], kl (scripts/eval_rc_kl.py: base vs rc [vs ctl] on
# the same sd_mixtest examples at recompute 0 / 0.05 / 0.1 / 0.2 -> results/rc_kl_<tag>), main (eval_recompute niah / ruler /
# qa / gsm8k / code for hyb, ours 10%, ours 20% with the rc projector -> results/rc_<tag>_main; the base projector's numbers
# are results/fused_<tag>_main on main). Steps are isolated; train is skipped when its DONE marker exists.
# Resume: --export=ALL,START_PAIR=<pair>[,START_STEP=<step>]   Subset: --export=ALL,PAIRS="06 17"
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"06 17 ll xll1q8 xq17ll8"}
START_PAIR=${START_PAIR:-}
START_STEP=${START_STEP:-}
STEPS=${STEPS:-4000}
RCF=${RCF:-0,0.1,0.1,0.2}
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
  CTL=$CK/rcctl_$TAG
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
      test)  python tests/test_rc_train.py $P ;;
      train) train $RC --rc_fracs $RCF ;;
      ctl)   [[ $PAIR == 06 ]] && train $CTL || echo "no control for $PAIR" ;;
      kl)    PJ=base=$P,rc=$RC/projector.pt; [ -f $CTL/DONE ] && PJ=$PJ,ctl=$CTL/projector.pt
             python scripts/eval_rc_kl.py --projectors $PJ --out_dir results/rc_kl_$TAG ;;
      main)  python scripts/eval_recompute.py --projector $RC/projector.pt --out_dir results/rc_${TAG}_main \
               --tasks niah,ruler,qa,gsm8k,code --configs hyb,attn_small:0.1:sent,attn_small:0.2:sent ;;
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
  for s in test train ctl kl main; do
    [[ $sstarted == 0 && $s == $START_STEP ]] && sstarted=1
    [[ $sstarted == 0 ]] && continue
    run_step $pair $s
    # a failed test or training makes the later steps of this pair meaningless
    rcs=$?
    [[ $rcs -ne 0 && $s == test && $pair == $(echo $PAIRS | cut -d" " -f1) ]] && { echo "test failed on the first pair: stopping"; exit 1; }
    if [[ $rcs -ne 0 && ( $s == test || $s == train ) ]]; then echo "skipping the rest of $pair"; break; fi
  done
done
step "finished; failed steps:${FAILED:- none}"
