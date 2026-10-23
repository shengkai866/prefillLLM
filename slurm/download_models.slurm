#!/bin/bash
#SBATCH --job-name=pf_dl_models
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# CPU-only (no GPU requested): download Qwen3-1.7B (~4 GB) and Qwen3-32B (~66 GB, bf16) into
# $PREFILL_ROOT/models/ for the larger model pairs. Re-running skips complete models and resumes partial ones.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/download_models.py Qwen/Qwen3-1.7B Qwen/Qwen3-32B
du -sh $PREFILL_ROOT/models/*
