#!/bin/bash
#SBATCH --job-name=pf_c2c
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-c2c/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Cache-to-Cache (C2C, Fu et al. ICLR'26, github.com/thu-nics/C2C) as "big model helps small model" baseline on the reverse
# pair Qwen3-8B (sharer / source) -> Qwen3-0.6B (receiver / decoder). Branch c2c-compare, worktree prefillLLM-c2c.
# Both models prefill the prompt; per 0.6B layer a C2CProjector adds a gated residual from [8B K/V | 0.6B K/V] onto the
# 0.6B's own K/V (prefill/c2c.py, CPU-verified == the official RosettaModel in tests/smoke_c2c.py).
#   EXP=e3 : C2C fuser architecture + OUR objective/data: KL to the 8B + 0.1 NLL on the 8B's own answers (sd_mix4k,
#            sd_mixlong), prompts <= 8k, 12000 steps x accum 4, no stage 1; C2C's lr 1e-4 / wd 0.01 / 10% warmup,
#            gate temperature annealed over all steps.                      -> ckpt c2c_e3_rev8to06, results/c2c_e3_*
#   EXP=e2 : C2C's OFFICIAL recipe (C2C_0.6+0.5.json with the 8B as sharer): NLL on OpenHermes-2.5 responses, max length
#            2048, macro batch 256 (4 x accum 64, one GPU instead of 8 x 4 x 8), lr 1e-4 linear, 10% warmup; scaled down
#            from 500k samples (1929 steps) to the first 130k (~500 steps; the authors report ~300 steps already match the
#            final checkpoint), gate annealing over 500 steps. Official trainer run as-is (scripts/c2c_official_train.py).
#                                                                            -> ckpt c2c_e2_rev8to06, results/c2c_e2_*
# Eval (both): KL to the 8B per data type + NIAH / RULER / QA / GSM8K / code + QuALITY, config "hyb" = the C2C fusion.
# 0.6B alone / 8B alone / our projector: prefillLLM-reverse results/rev8to06_*, prefillLLM-shortalign results/sa1k_rev8to06_*.
# Restart: sbatch --export=ALL,EXP=e3,START_AT=<smoke|train|eval_v2|main|quality> slurm/run_c2c.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-c2c || exit 1
M=$PREFILL_ROOT/models
export PREFILL_SMALL_DIR=$M/Qwen3-8B PREFILL_LARGE_DIR=$M/Qwen3-0.6B PREFILL_TEACHER=small PREFILL_FOLD=eager
export PREFILL_C2C_DIR=$PREFILL_ROOT/ext/C2C
EXP=${EXP:-e3}
TAG=c2c_${EXP}_rev8to06
CK=$PREFILL_ROOT/checkpoints/$TAG
R=results/$TAG
echo "C2C $EXP: worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD); C2C clone $(git -C $PREFILL_C2C_DIR rev-parse --short HEAD)"
START_AT=${START_AT:-smoke}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$3"; echo "!!!!!!!! FAILED: $*"; }; }

if [[ $EXP == e3 ]]; then
  P=$CK/projector.pt
  if should smoke; then
    set -e
    step "smoke: CPU equivalence test, a few GPU training steps, tiny evals"
    python tests/smoke_c2c.py
    rm -rf ${CK}_smoke
    python scripts/train_stage2.py --init c2c --out_dir ${CK}_smoke --max_steps 4 --accum 2 --lr 1e-4 --warmup 1 \
      --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
    python scripts/evaluate_v2.py --projectors e3=${CK}_smoke/projector.pt --baselines big --tasks kl --n_kl 1 --out_dir ${CK}_smoke/e_v2
    python scripts/eval_recompute.py --projector ${CK}_smoke/projector.pt --out_dir ${CK}_smoke/e_main --tasks niah,gsm8k,quality \
      --configs hyb --n_gsm8k 2 --n_quality 3
    set +e
  fi
  if should train; then
    step "train: C2C fuser, KL to the 8B, mixed data, prompts <= 8k, 12000 steps"
    python scripts/train_stage2.py --init c2c --max_steps 12000 --accum 4 --lr 1e-4 --warmup 1200 --wd 0.01 \
      --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
      --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $CK || exit 1
  fi
  CURVE="kl,curve --curve_runs $TAG=$CK"
else
  P=$CK/final
  if should smoke; then
    step "smoke: CPU equivalence test"
    python tests/smoke_c2c.py || exit 1
  fi
  if should train; then
    step "train: official C2C trainer, OpenHermes-2.5 first 130k, ~500 steps of 256"
    mkdir -p $CK
    cat > $CK/recipe.json <<EOF
{
  "model": {
    "base_model": "$M/Qwen3-0.6B",
    "teacher_model": "$M/Qwen3-8B",
    "is_do_alignment": false,
    "alignment_strategy": "first",
    "projector": {"type": "C2CProjector", "params": {"hidden_dim": 1024, "intermediate_dim": 1024, "num_layers": 3,
      "dropout": 0.1, "initial_temperature": 1.0, "final_temperature": 0.001, "anneal_steps": 500}},
    "mapping": "last_aligned"
  },
  "training": {"learning_rate": 1e-4, "weight_decay": 0.01, "num_epochs": 1, "max_length": 2048, "device": "cuda",
    "scheduler_type": "linear", "warmup_ratio": 0.1, "max_grad_norm": 1.0, "gradient_accumulation_steps": 64,
    "per_device_train_batch_size": 4, "num_processes": 1, "freeze": ["teacher", "base"], "seed": 42},
  "output": {"output_dir": "$CK", "save_steps": 250, "eval_steps": 100,
    "wandb_config": {"project": "Rosetta", "mode": "disabled", "entity": null, "run_name": "$TAG"}},
  "data": {"type": "OpenHermesChatDataset", "kwargs": {"split": "train", "max_word_count": 2048, "num_samples": 130000},
    "train_ratio": 0.99}
}
EOF
    python scripts/c2c_official_train.py --config $CK/recipe.json || exit 1
    # the trainer also dumps the whole RosettaModel state dict (frozen 8B + 0.6B, ~17 GB) next to every checkpoint
    rm -f $CK/final/model.pt $CK/checkpoint-*/model.pt
  fi
  CURVE="kl"
fi

if should eval_v2; then
  step "eval: KL vs the 8B per data type (C2C fusion)"
  ev python scripts/evaluate_v2.py --projectors $TAG=$P --baselines big --tasks $CURVE --out_dir ${R}_v2
fi
if should main; then
  step "eval main: NIAH / RULER / QA / GSM8K / code (hyb = 8B + 0.6B prefill, C2C fusion, 0.6B decodes)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs hyb
fi
if should quality; then
  step "eval QuALITY (single question)"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_quality --tasks quality --configs hyb
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
