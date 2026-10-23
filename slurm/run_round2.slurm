#!/bin/bash
#SBATCH --job-name=pf_round2
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=4-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Round 2, one GPU, strictly sequential:
#   data pools -> 8B self-distillation -> three stage-2 runs from the same round-1 stage-1 init -> evaluation -> figures
#   runs: mix4k (mixed data, prompts <= 4k) | ctrl4k (web only, same steps) | mix8k (mixed data, prompts <= 8k)
# Data/weights go to scratch; results/round2/{summary.md,results.json,figures/} go to this project directory.
# Restart from a step:  sbatch --export=ALL,START_AT=train_ctrl4k slurm/run_round2.slurm
#   steps: prep gen_test gen_mix4k train_mix4k gen_web4k train_ctrl4k gen_mixlong train_mix8k eval plot
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
START_AT=${START_AT:-prep}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }

INIT=$PREFILL_ROOT/checkpoints/stage1/projector.pt      # round-1 stage-1 projector, shared init of all three runs
STEPS=12000                                              # 4x the round-1 stage-2 steps
COMMON="--init $INIT --max_steps $STEPS --accum 4 --lr 5e-5 --warmup 100 --val_every 500 --val_per_type 2 \
        --save_every 500 --save_step_every 2000 --time_budget_hours 40"

if should prep;         then step "prep: data pools";               python scripts/prepare_mix.py; fi
if should gen_test;     then step "gen: held-out 8B answers";       python scripts/gen_selfdistill.py --pool mixtest --out sd_mixtest --greedy; fi
if should gen_mix4k;    then step "gen: mixed 4k";                  python scripts/gen_selfdistill.py --pool mix4k_train --out sd_mix4k; fi
if should train_mix4k;  then step "train: mixed data, 4k";          python scripts/train_stage2.py $COMMON --sd sd_mix4k --max_prompt 3968 --out_dir $PREFILL_ROOT/checkpoints/r2_mix4k; fi
if should gen_web4k;    then step "gen: web only 4k (control)";     python scripts/gen_selfdistill.py --pool web4k_train --out sd_web4k; fi
if should train_ctrl4k; then step "train: control, web only, 4k";   python scripts/train_stage2.py $COMMON --sd sd_web4k --max_prompt 3968 --out_dir $PREFILL_ROOT/checkpoints/r2_ctrl4k; fi
if should gen_mixlong;  then step "gen: mixed 4k-8k";               python scripts/gen_selfdistill.py --pool mixlong_train --out sd_mixlong; fi
if should train_mix8k;  then step "train: mixed data, 8k";          python scripts/train_stage2.py $COMMON --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $PREFILL_ROOT/checkpoints/r2_mix8k; fi
if should eval;         then step "eval";                           python scripts/evaluate_v2.py --out_dir results/round2; fi
if should plot;         then step "plot";                           python scripts/plot_round2.py results/round2; cat results/round2/summary.md; fi
step "all done"
