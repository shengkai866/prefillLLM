#!/bin/bash
#SBATCH --job-name=pf_r5c
#SBATCH --account=drjieliu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Resume the three round-5 runs that were stopped on 2026-10-22, ONE job at a time: each stage, as soon as it starts,
# queues the next stage with --dependency=afterany on itself (so the chain goes on even if a stage fails).
#   stage 1  normA   1.7B->8B stage 2 (in_norm), resume at step 5500 of 12000              [1 GPU, L40S or H100]
#   stage 2  normB   same + attention-output loss, resume at step 4000 of 12000           [1 H100: OOMs on L40S]
#   stage 3  more32  1.7B->32B, resume at step 5500 of 24000                              [2 H100]
# Each stage then runs the same evals as slurm/run_r5.slurm (eval_rc, eval_v2).
# Submit the whole chain:      bash slurm/run_r5_resume.slurm        (or: bash slurm/run_r5_resume.slurm <stage>)
# Restart one stage at a step: bash slurm/run_r5_resume.slurm 2 eval_rc   (steps: train eval_rc eval_v2; chain continues)
# Stop the whole chain:        scancel -u shengkai -n pf_r5c
SELF=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/run_r5_resume.slurm
NSTAGES=3

submit_stage() {  # $1 stage, $2 START_AT, $3 extra sbatch args (e.g. a dependency)
  local res
  case $1 in
    1) res="--partition=drjieliu-l40s,drjieliu-h100 --gres=gpu:1 --cpus-per-task=4 --mem=160G --time=2-00:00:00" ;;
    2) res="--partition=drjieliu-h100 --gres=gpu:1 --cpus-per-task=4 --mem=160G --time=2-00:00:00" ;;
    3) res="--partition=drjieliu-h100 --gres=gpu:2 --cpus-per-task=6 --mem=200G --time=3-00:00:00" ;;
  esac
  sbatch $res $3 --export=ALL,STAGE=$1,START_AT=$2 $SELF
}

if [[ -z $SLURM_JOB_ID ]]; then  # called with bash on the login node: submit the first stage
  submit_stage ${1:-1} ${2:-train}
  exit
fi

set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
STAGE=${STAGE:-1}
if (( STAGE < NSTAGES )); then
  echo "[chain] stage $STAGE running; queueing stage $((STAGE + 1)) after job $SLURM_JOB_ID"
  submit_stage $((STAGE + 1)) train "--dependency=afterany:$SLURM_JOB_ID"
fi

CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
START_AT=${START_AT:-train}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
S2COMMON="--accum 4 --warmup 100 --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 60 \
          --sd sd_mix4k,sd_mixlong --max_prompt 8064"
SPLIT=""
case $STAGE in
  1|2)
    JOB=$([[ $STAGE == 1 ]] && echo normA || echo normB)
    export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
    OUT=$CK/r5_s17_$JOB ; RES=results/r5_s17_$JOB ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
    FROM=$([[ $JOB == normA ]] && echo 5500 || echo 4000)
    EXTRA="" ; [[ $JOB == normB ]] && EXTRA="--lambda_attn 1.0 --attn_layers 4"
    ARGS="--max_steps 12000 --lr 5e-5 $S2COMMON $EXTRA" ;;
  3)
    JOB=more32
    export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B ; SPLIT="24GiB,76GiB"
    OUT=$CK/r5_more_17_32 ; RES=results/r5_more_17_32 ; RC_CFGS=hyb,attn_small:0.05:sent,attn_small:0.1:sent
    FROM=5500
    ARGS="--max_steps 24000 --lr 3e-5 --curriculum_frac 0 $S2COMMON" ;;
  *) echo "STAGE must be 1..$NSTAGES"; exit 1 ;;
esac

if should train; then
  step "train: $JOB, resume from step $FROM"
  # projector.pt gets overwritten while training, so resume from a frozen copy of the step-$FROM checkpoint
  [[ -f $OUT/projector_resume$FROM.pt ]] || cp $OUT/projector.pt $OUT/projector_resume$FROM.pt
  TRAIN="python scripts/train_stage2.py --init $OUT/projector_resume$FROM.pt --start_step $FROM $ARGS --out_dir $OUT"
  echo "$TRAIN"
  PREFILL_LARGE_MAXMEM=$SPLIT $TRAIN
fi
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
