#!/bin/bash
#SBATCH --job-name=pf3_stage2
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
# Stage 2: self-distillation data generation by Qwen3-8B, then end-to-end KL distillation through the frozen 8B.
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/train_stage2.py --n_gen 2400 --max_steps 3000 --accum 4 --lr 5e-5 --time_budget_hours 22
