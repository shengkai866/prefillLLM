#!/bin/bash
#SBATCH --job-name=pf_r4_more
#SBATCH --account=drjieliu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=2-12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Round 4: is training amount the bottleneck? Continue each round-2/3 stage-2 projector for another 12000 steps with the
# SAME data (no length curriculum, since the runs already reached 8k prompts), then evaluate.
# Learning rate: runs whose projector has already learned retrieval (06_8, 17_8) continue from near the end of the
# original schedule (cosine 1e-5 -> 1e-6): a first attempt that re-warmed to 5e-5 (jobs 27907255/6) knocked out the learned
# retrieval within 500-1500 steps. 17_32 never learned retrieval, so it keeps 5e-5.
#   PAIR=06_8   0.6B -> 8B   from checkpoints/r2_mix8k       (1 GPU, H100 or L40S)
#   PAIR=17_8   1.7B -> 8B   from checkpoints/s17_mix8k      (1 GPU, H100 or L40S)
#   PAIR=17_32  1.7B -> 32B  from checkpoints/s17to32_mix8k  (2 H100 for training, eval on one)
# Submit:  sbatch --partition=drjieliu-h100,drjieliu-l40s --gres=gpu:1 --export=ALL,PAIR=06_8 slurm/run_r4_more_steps.slurm
#          sbatch --partition=drjieliu-h100 --gres=gpu:2 --export=ALL,PAIR=17_32 slurm/run_r4_more_steps.slurm
# Steps: train -> eval_rc (RULER / QA / NIAH, with and without recompute) -> eval_v2 (KL per type, GSM8K, code, and
#        KL + retrieval accuracy at every saved checkpoint of the original AND the continued run)
# Restart from a step:  --export=ALL,PAIR=...,START_AT=eval_rc
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
SPLIT=""
case "$PAIR" in
  06_8)  ORIG=$CK/r2_mix8k ;      LR=1e-5 ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05 ;;
  17_8)  ORIG=$CK/s17_mix8k ;     LR=1e-5 ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
         export PREFILL_SMALL_DIR=$M/Qwen3-1.7B ;;
  17_32) ORIG=$CK/s17to32_mix8k ; LR=5e-5 ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent
         export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B ; SPLIT="24GiB,76GiB" ;;
  *) echo "set PAIR=06_8|17_8|17_32"; exit 1 ;;
esac
OUT=$CK/r4_more_${PAIR}
RES=results/r4_more_${PAIR}
START_AT=${START_AT:-train}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }

if should train; then
  step "train: $PAIR, +12000 stage-2 steps from $ORIG/projector.pt, peak lr $LR"
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage2.py --init $ORIG/projector.pt --max_steps 12000 --accum 4 --lr $LR \
    --warmup 100 --curriculum_frac 0 --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 \
    --time_budget_hours 40 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $OUT
fi
if should eval_rc; then
  step "eval: RULER / QA / NIAH with and without selective recompute"
  python scripts/eval_recompute.py --projector $OUT/projector.pt --out_dir $RES --tasks niah,ruler,qa --configs $RC_CFGS
fi
if should eval_v2; then
  step "eval: KL per type, GSM8K, code; KL + retrieval at every checkpoint (orig = steps 0-12000, cont = 12000-24000)"
  python scripts/evaluate_v2.py --projectors cont=$OUT/projector.pt --baselines "" --tasks kl,gsm8k,code,curve \
    --curve_runs orig=$ORIG,cont=$OUT --out_dir ${RES}_v2
fi
step "all done"
