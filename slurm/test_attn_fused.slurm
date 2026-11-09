#!/bin/bash
#SBATCH --job-name=pf_attnk
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
# Unit test + micro-benchmark of the position-causal attention kernel (prefill/attn_fused.py).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM || exit 1
python tests/test_attn_fused.py --lengths ${LENGTHS:-4096,8192,16384,32768} --reps 5
