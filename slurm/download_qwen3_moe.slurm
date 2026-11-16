#!/bin/bash
#SBATCH --job-name=pf_dl_moe
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
# CPU-only (no GPU requested): download Qwen3-30B-A3B (MoE, 30B total / 3B active, ~61 GB bf16, 16 shards) into
# $PREFILL_ROOT/models/Qwen3-30B-A3B on scratch. There is no "Qwen3-32B-A3B" on the Hub; this is the Qwen3 MoE of the
# same release as Qwen3-0.6B/1.7B/4B/8B/32B (hybrid thinking, same tokenizer). Re-running skips a complete model and
# resumes a partial one.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/download_models.py Qwen/Qwen3-30B-A3B
du -sh $PREFILL_ROOT/models/Qwen3-30B-A3B
