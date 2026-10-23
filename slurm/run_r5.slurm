#!/bin/bash
#SBATCH --job-name=pf_r5
#SBATCH --account=drjieliu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Round 5. Pick the job with JOB=..., GPUs/partition on the sbatch line:
#   more32  1.7B->32B: continue the round-4 projector (24000 steps so far) for 24000 more, cosine 3e-5 -> 3e-6
#           (it was still learning retrieval at the end of round 4).                          [2 H100]
#   s1norm  1.7B->8B stage 1 from scratch with RMS-normalised small-model inputs (--in_norm).   [1 GPU]
#   normA   1.7B->8B stage 2 from s1norm, round-2 mix8k recipe (only change vs round 3a: in_norm).   [1 GPU, after s1norm]
#   normB   same as normA + stage-2 attention-output matching loss (lambda 1, 4 layers from the middle third on).
#   eval06  0.6B->8B round-2 mix8k projector: GSM8K and code next-line with the recompute configs (for the 0.6B report).
# Each training job then evaluates: eval_recompute (RULER / QA / NIAH, with and without recompute) and evaluate_v2
# (KL per type, GSM8K, code, KL + retrieval accuracy at every saved checkpoint).
# Restart a job from a step:  --export=ALL,JOB=normA,START_AT=eval_rc   (steps: train eval_rc eval_v2)
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
START_AT=${START_AT:-train}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
S2COMMON="--accum 4 --warmup 100 --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 60 \
          --sd sd_mix4k,sd_mixlong --max_prompt 8064"
SPLIT=""
case "$JOB" in
  more32)
    export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B ; SPLIT="24GiB,76GiB"
    OUT=$CK/r5_more_17_32 ; RES=results/r5_more_17_32 ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent
    TRAIN="python scripts/train_stage2.py --init $CK/r4_more_17_32/projector.pt --max_steps 24000 --lr 3e-5 --curriculum_frac 0 $S2COMMON --out_dir $OUT" ;;
  s1norm)
    export PREFILL_SMALL_DIR=$M/Qwen3-1.7B ; OUT=$CK/r5_s17norm_stage1
    TRAIN="python scripts/train_stage1.py --in_norm --out_dir $OUT --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 --resume $OUT/projector.pt" ;;
  normA|normB)
    export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
    OUT=$CK/r5_s17_$JOB ; RES=results/r5_s17_$JOB ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
    EXTRA="" ; [[ $JOB == normB ]] && EXTRA="--lambda_attn 1.0 --attn_layers 4"
    TRAIN="python scripts/train_stage2.py --init $CK/r5_s17norm_stage1/projector.pt --max_steps 12000 --lr 5e-5 $S2COMMON $EXTRA --out_dir $OUT" ;;
  eval06)
    START_AT=eval_only ;;
  *) echo "set JOB=more32|s1norm|normA|normB|eval06"; exit 1 ;;
esac

if [[ $JOB == eval06 ]]; then
  step "eval: 0.6B->8B (round-2 mix8k projector) GSM8K + code with recompute configs"
  python scripts/eval_recompute.py --projector $CK/r2_mix8k/projector.pt --out_dir results/r5_eval06 --tasks gsm8k,code \
    --configs big,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
  step "all done"; exit 0
fi
if should train; then
  step "train: $JOB"
  echo "$TRAIN"
  PREFILL_LARGE_MAXMEM=$SPLIT $TRAIN
fi
[[ $JOB == s1norm ]] && { step "all done"; exit 0; }
if should eval_rc; then
  step "eval: RULER / QA / NIAH with and without selective recompute"
  python scripts/eval_recompute.py --projector $OUT/projector.pt --out_dir $RES --tasks niah,ruler,qa --configs $RC_CFGS
fi
if should eval_v2; then
  step "eval: KL per type, GSM8K, code; KL + retrieval at every checkpoint"
  python scripts/evaluate_v2.py --projectors $JOB=$OUT/projector.pt --baselines "" --tasks kl,gsm8k,code,curve \
    --curve_runs $JOB=$OUT --out_dir ${RES}_v2
fi
step "all done"
