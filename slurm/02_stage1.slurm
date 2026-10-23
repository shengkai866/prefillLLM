#!/bin/bash
#SBATCH --job-name=pf2_stage1
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Stage 1: representation alignment (cos + normalised MSE + attention-output matching). Single GPU.
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/train_stage1.py --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 \
  --resume $PREFILL_ROOT/checkpoints/stage1/projector.pt
