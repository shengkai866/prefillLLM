#!/bin/bash
#SBATCH --job-name=pf_dl_llama
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
# CPU-only (no GPU requested): Llama-3.2-1B and Llama-3.1-8B, Instruct and base (~2.5 + 16 GB each), into
# $PREFILL_ROOT/models/ on scratch. The official meta-llama repos are gated and there is no HF token here, so the public
# unsloth mirrors are used (same bf16 safetensors). Re-running skips complete models and resumes partial ones.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/download_models.py unsloth/Llama-3.2-1B-Instruct unsloth/Llama-3.1-8B-Instruct unsloth/Llama-3.2-1B unsloth/Llama-3.1-8B
du -sh $PREFILL_ROOT/models/Llama-*
