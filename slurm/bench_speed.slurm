#!/bin/bash
#SBATCH --job-name=pf_speed
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Speed benchmark of the optimised hybrid prefill with the trained stage-2 projector (H100 for comparability with
# the first run), plus NIAH / per-layer diagnostics of where prompt details are lost. No training.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/bench_speed.py --sections check,speed,niah,probe --out_dir results/speed_v2
