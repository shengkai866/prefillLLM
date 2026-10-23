#!/bin/bash
#SBATCH --job-name=pf_all
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# One job, run sequentially: data check -> stage1 -> stage2 -> eval. Stops at the first failing step.
# To skip finished steps:  sbatch --export=ALL,START_AT=stage2 slurm/run_all.slurm   (data|stage1|stage2|eval)
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
START_AT=${START_AT:-data}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }

if should data; then
  step "data (idempotent: skips anything already downloaded)"
  python scripts/download_and_prepare.py --n_train 12000 --n_test 300
fi
if should stage1; then
  step "stage1: representation alignment"
  python scripts/train_stage1.py --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 \
    --resume "$PREFILL_ROOT/checkpoints/stage1/projector.pt"
fi
if should stage2; then
  step "stage2: self-distillation + end-to-end KL"
  python scripts/train_stage2.py --n_gen 2400 --max_steps 3000 --accum 4 --lr 5e-5 --time_budget_hours 22
fi
if should eval; then
  step "eval: big vs small vs hybrid"
  python scripts/evaluate.py --tasks speed,ppl,oracle,niah,gsm8k --out_dir results/main
  echo "== summary =="; cat results/main/summary.md
fi
step "all done"
