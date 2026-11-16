#!/bin/bash
#SBATCH --job-name=pf_tmoe
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-tokmoe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Token-routed MoE projector (prefill/projector.ProjectorTokMoE; branch tok-moe, worktree prefillLLM-tokmoe) on the
# reverse pair Qwen3-30B-A3B (prefills, teacher) -> Qwen3-0.6B (decodes). Same environment, data tree (data_a3b) and
# stage-2 recipe as slurm/run_a3b.slurm METHOD=rev, which is the baseline: results/rev30a3to06_* in prefillLLM-a3b.
#   INIT=conv (default) : the baseline's stage-1 checkpoint ($CK/rev30a3to06_stage1, plain Projector, 8000 steps) is
#                         converted exactly (every expert = its trunk), then the usual stage 2 (KL to the 30B-A3B,
#                         mix8k, 12000 steps). Same stage 1, same stage-2 budget as the baseline: the only difference is
#                         the routed experts.                                          -> ckpt/results tmoe<E>_rev30a3to06
#   INIT=scratch        : stage 1 (8000 steps) already with the MoE projector, then stage 2 -> tmoe<E>s_rev30a3to06
#   INIT=smoke          : CPU test, a few steps of both paths, one forward/backward on the longest 8k training prompt
#                         (peak GPU memory), tiny evals                                -> ckpt tmoe_smoke
# EXPERTS (8), TOPK (2), BAND (0 = pure token routing; e.g. 2 = neighbouring layers start on the same experts, tag
# suffix _b<BAND>), LAMBDA_BAL (0.01). Needs METHOD=gen and (INIT=conv) METHOD=rev stage 1 of run_a3b.slurm.
# Restart: sbatch --export=ALL,INIT=<i>,START_AT=<stage1|stage2|eval_v2|main|quality> slurm/run_tmoe.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-tokmoe || exit 1
M=$PREFILL_ROOT/models
export PREFILL_SMALL_DIR=$M/Qwen3-30B-A3B PREFILL_LARGE_DIR=$M/Qwen3-0.6B PREFILL_TEACHER=small
export PREFILL_FOLD=eager
export PREFILL_SMALL_MAXMEM=${PREFILL_SMALL_MAXMEM:-22GiB,44GiB}
INIT=${INIT:-conv}
EXPERTS=${EXPERTS:-8}; TOPK=${TOPK:-2}; BAND=${BAND:-0}; LAMBDA_BAL=${LAMBDA_BAL:-0.01}
PT=rev30a3to06
CK=$PREFILL_ROOT/checkpoints
TM="--tmoe $EXPERTS --tmoe_topk $TOPK --tmoe_band $BAND --lambda_bal $LAMBDA_BAL"
SFX=$([[ $TOPK == 2 ]] || echo "k$TOPK")$([[ $BAND == 0 ]] || echo "_b$BAND")
case $INIT in
  smoke)   TAG=tmoe_smoke ;;
  conv)    TAG=tmoe${EXPERTS}${SFX}_$PT ;;
  scratch) TAG=tmoe${EXPERTS}s${SFX}_$PT ;;
  *) echo "unknown INIT $INIT"; exit 1 ;;
esac
R=results/$TAG
DA=$PREFILL_ROOT/data_a3b
[[ $INIT != smoke ]] && export PREFILL_DATA_DIR=$DA
echo "INIT=$INIT tag $TAG ($TM); worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD);" \
     "source $PREFILL_SMALL_DIR (split $PREFILL_SMALL_MAXMEM) -> decoder $PREFILL_LARGE_DIR, teacher = source"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
START_AT=${START_AT:-$([[ $INIT == scratch ]] && echo stage1 || echo stage2)}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$3"; echo "!!!!!!!! FAILED: $*"; }; }  # isolated eval step

if [[ $INIT == smoke ]]; then
  set -e
  SM=$CK/$TAG
  rm -rf $SM
  step "smoke 1/4: CPU test"
  CUDA_VISIBLE_DEVICES= python tests/test_tokmoe.py
  step "smoke 2/4: scratch path, stage 1 with the MoE projector, a few steps"
  python scripts/train_stage1.py --in_norm $TM --out_dir $SM/s1moe --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  step "smoke 3/4: conv path, plain stage 1 -> converted -> stage 2 a few steps; longest 8k prompt first (peak memory)"
  python scripts/train_stage1.py --in_norm --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  python scripts/train_stage2.py --init $SM/s1/projector.pt $TM --probe_longest --out_dir $SM/s2 --max_steps 4 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  python scripts/train_stage2.py --init $SM/s1moe/projector.pt --out_dir $SM/s2moe --max_steps 2 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  step "smoke 4/4: tiny evals of the MoE checkpoint (TokMoEKV inference path)"
  python scripts/evaluate_v2.py --projectors tmoe=$SM/s2/projector.pt --baselines big --tasks kl --n_kl 1 --out_dir $SM/e_v2
  python scripts/eval_recompute.py --projector $SM/s2/projector.pt --out_dir $SM/e_main --tasks niah,gsm8k,quality \
    --configs hyb --n_gsm8k 2 --n_quality 3
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  set +e
  step "smoke passed"
  exit 0
fi

for n in sd_mixtest sd_mix4k sd_mixlong; do
  [ -e $DA/$n/meta.jsonl ] || { echo "missing $DA/$n (run_a3b.slurm METHOD=gen first)"; exit 1; }
done
S2=$CK/${TAG}_mix8k
P=$S2/projector.pt
if [[ $INIT == conv ]]; then
  S1=$CK/${PT}_stage1  # the baseline's stage 1 (plain Projector)
  [ -e $S1/projector.pt ] || { echo "missing $S1/projector.pt (run_a3b.slurm METHOD=rev stage 1 first)"; exit 1; }
  S2ARGS="$TM"
else
  S1=$CK/${TAG}_stage1
  S2ARGS="--lambda_bal $LAMBDA_BAL"  # the stage-1 checkpoint is a MoE projector already
  if should stage1; then
    step "stage1: 30B-A3B hidden states -> 0.6B residual streams, in_norm, MoE projector ($TM), 8000 steps"
    python scripts/train_stage1.py --in_norm $TM --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 20 \
      --resume $S1/projector.pt || exit 1
  fi
fi
if should stage2; then
  step "stage2: KL to the 30B-A3B, mixed data, prompts <= 8k (mix8k recipe), 12000 steps; init $S1"
  python scripts/train_stage2.py --init $S1/projector.pt $S2ARGS --probe_longest --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2 || exit 1
fi
if should eval_v2; then
  step "eval: KL vs the 30B-A3B per data type"
  ev python scripts/evaluate_v2.py --projectors $TAG=$P --baselines big --tasks kl,curve --curve_runs $TAG=$S2 --out_dir ${R}_v2
fi
if should main; then
  step "eval main: NIAH / RULER / QA / GSM8K / code (baselines big / small / plain hyb: results/rev30a3to06_main in prefillLLM-a3b)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs hyb
fi
if should quality; then
  step "eval QuALITY (single question)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_quality --tasks quality --configs hyb
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
