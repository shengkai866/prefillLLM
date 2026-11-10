#!/bin/bash
#SBATCH --job-name=pf_rev8to06
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-reverse/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# REVERSE pair (branch reverse-prefill, worktree prefillLLM-reverse): Qwen3-8B prefills, a projector maps its hidden
# states to Qwen3-0.6B's KV cache, Qwen3-0.6B decodes. In code terms the 8B is the "small" model (source of hidden states)
# and the 0.6B is the "large" model (owner of the cache); PREFILL_TEACHER=small makes the 8B the stage-2 KL teacher, so
# the 0.6B should answer like the 8B. Pure projector only: no recompute configs (the user's first look at this case).
# Same recipe as the forward pairs: stage 1 in_norm 8000 steps (hidden-state alignment to the 0.6B's residual streams),
# stage 2 mix8k 12000 steps on the 8B's own self-distillation answers (sd_mix4k, sd_mixlong of the Qwen data tree).
#   smoke    : a few training steps + tiny eval runs
#   stage1   : -> $PREFILL_ROOT/checkpoints/rev8to06_stage1
#   stage2   : -> $PREFILL_ROOT/checkpoints/rev8to06_mix8k
#   eval_v2  : KL vs the 8B per data type (0.6B alone and ours), KL + retrieval at every checkpoint -> results/rev8to06_v2
#   main     : NIAH / RULER / QA / GSM8K / code, configs big (= 0.6B alone), small (= 8B alone), hyb (ours)
#   quality  : QuALITY single question, same three configs                                   -> results/rev8to06_quality
# Results go to this worktree's results/ (prefill.config.PROJECT_DIR). Restart: sbatch --export=ALL,START_AT=<step> ...
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-reverse || exit 1
# PAIR (sbatch --export=ALL,PAIR=...): q8to06 (default) | q8to17 (Qwen3-8B -> Qwen3-1.7B) | ll8to1 (Llama-3.1-8B-Instruct ->
# Llama-3.2-1B-Instruct, Llama data tree whose sd_* answers are the Llama-8B's) | xll8toq17 (Llama-3.1-8B -> Qwen3-1.7B)
# | xq8toll1 (Qwen3-8B -> Llama-3.2-1B): cross-family, see below.
PAIR=${PAIR:-q8to06}
M=$PREFILL_ROOT/models
case $PAIR in
  q8to06) export PREFILL_SMALL_DIR=$M/Qwen3-8B PREFILL_LARGE_DIR=$M/Qwen3-0.6B; TAG=rev8to06 ;;
  q8to17) export PREFILL_SMALL_DIR=$M/Qwen3-8B PREFILL_LARGE_DIR=$M/Qwen3-1.7B; TAG=rev8to17 ;;
  ll8to1) source slurm/llama_env.sh
          export PREFILL_SMALL_DIR=$M/Llama-3.1-8B-Instruct PREFILL_LARGE_DIR=$M/Llama-3.2-1B-Instruct; TAG=revll8to1 ;;
  # cross-family: the decoder keeps its own tokens / pools, the 8B reads the C2C-mapped sequence; stage-2 KL over the
  # ~110k tokens both vocabularies share (prefill/tokalign.shared_vocab); training data = the 8B's own answers
  # (its family's sd_* stores) converted to the decoder's tokens and chat template in a data tree of its own
  xll8toq17) export PREFILL_SMALL_DIR=$M/Llama-3.1-8B-Instruct PREFILL_LARGE_DIR=$M/Qwen3-1.7B; TAG=revll8toq17
          CROSS=1; BASE=$PREFILL_ROOT/data; SRC=$PREFILL_ROOT/data_llama ;;
  xq8toll1) export PREFILL_SMALL_DIR=$M/Qwen3-8B PREFILL_LARGE_DIR=$M/Llama-3.2-1B-Instruct; TAG=rev8toll1
          CROSS=1; BASE=$PREFILL_ROOT/data_llama; SRC=$PREFILL_ROOT/data ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
export PREFILL_TEACHER=small
CROSS=${CROSS:-0}
[[ $CROSS == 1 ]] && export PREFILL_DATA_DIR=$PREFILL_ROOT/data_$TAG
export PREFILL_FOLD=eager  # the fused fold kernel is only validated for forward pairs; speed is not measured here
echo "worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD); PAIR=$PAIR source $PREFILL_SMALL_DIR -> target $PREFILL_LARGE_DIR, teacher = source, data ${PREFILL_DATA_DIR:-default}"
START_AT=${START_AT:-$([[ $CROSS == 1 ]] && echo convert || echo smoke)}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
CK=$PREFILL_ROOT/checkpoints
SM=$CK/${TAG}_smoke
S1=$CK/${TAG}_stage1
S2=$CK/${TAG}_mix8k
P=$S2/projector.pt
R=results/$TAG
CFG=big,small,hyb
[[ $CROSS == 1 ]] && CFG=big,hyb  # the 8B alone is in its own family's results (it cannot decode the decoder's tokens)
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$3"; echo "!!!!!!!! FAILED: $*"; }; }  # isolated eval step

if [[ $CROSS == 1 ]] && should convert; then
  step "convert: data tree $PREFILL_DATA_DIR = links into $BASE + the 8B's sd_* stores from $SRC in the decoder's tokens"
  mkdir -p $PREFILL_DATA_DIR
  for e in $BASE/*; do n=$(basename $e); case $n in sd_mix4k|sd_mixlong|sd_mixtest) ;; *) [ -e $PREFILL_DATA_DIR/$n ] || ln -s $e $PREFILL_DATA_DIR/$n ;; esac; done
  [ -e $PREFILL_DATA_DIR/sd_mixtest/meta.jsonl ] || python scripts/convert_sd_cross.py --src_dir $SRC --dst_dir $PREFILL_DATA_DIR \
    --src_tok $PREFILL_SMALL_DIR --dst_tok $PREFILL_LARGE_DIR || exit 1
fi
if should smoke; then
  set -e
  step "smoke: a few GPU training steps and tiny eval runs"
  rm -rf $SM
  python scripts/train_stage1.py --in_norm --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/s2 --max_steps 4 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  Q=$SM/s2/projector.pt
  python scripts/evaluate_v2.py --projectors rev=$Q --baselines big --tasks kl --n_kl 1 --out_dir $SM/e_v2
  python scripts/eval_recompute.py --projector $Q --out_dir $SM/e_main --tasks niah,gsm8k,quality --configs $CFG --n_gsm8k 2 --n_quality 3
  grep -h "KL" -A4 $SM/e_v2/summary.md | head -6 || true
  set +e
fi
if should stage1; then
  step "stage1: 8B hidden states -> 0.6B residual streams, in_norm"
  python scripts/train_stage1.py --in_norm --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 \
    --resume $S1/projector.pt || exit 1
fi
if should stage2; then
  step "stage2: KL to the 8B, mixed data, prompts <= 8k (mix8k recipe)"
  python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2 || exit 1
fi
if should eval_v2; then
  step "eval: KL vs the 8B per data type (0.6B alone and ours); KL + retrieval at every checkpoint"
  ev python scripts/evaluate_v2.py --projectors $TAG=$P --baselines big --tasks kl,curve --curve_runs $TAG=$S2 --out_dir ${R}_v2
fi
if should main; then
  step "eval main: NIAH / RULER / QA / GSM8K / code (0.6B alone, 8B alone, 8B prefill -> projector -> 0.6B decode)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs $CFG
fi
if should quality; then
  step "eval QuALITY (single question)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_quality --tasks quality --configs $CFG
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
