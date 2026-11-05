#!/bin/bash
#SBATCH --job-name=pf_moediag
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch moe-router: routing diagnostic over the (layer x token) grid for a MoE-style projector (scripts/diag_moe.py).
# 0.6B -> 8B, projector r2_mix8k, sd_mixtest 6 types x 4k/8k x N examples. Results: results/moe_diag/ (in this worktree).
# Restart resumes from per_example.jsonl.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/env.sh
python scripts/diag_moe.py --n_per_cell ${N:-10} --out_dir results/${OUT:-moe_diag}
