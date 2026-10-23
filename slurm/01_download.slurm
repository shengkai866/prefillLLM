#!/bin/bash
#SBATCH --job-name=pf1_download
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# CPU-only: download Qwen3-0.6B + Qwen3-8B, stream FineWeb-Edu + SlimPajama (tokenised subsets), GSM8K.
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/download_and_prepare.py --n_train 12000 --n_test 300
