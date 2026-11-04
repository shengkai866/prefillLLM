#!/bin/bash
#SBATCH --job-name=pf_foldk
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold/logs/%x-%j.out
# Unit test + micro-benchmark of the fused fold kernel (fold-kernel worktree; the main checkout is untouched).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-fold || exit 1
export TRITON_CACHE_DIR=$PREFILL_ROOT/triton_cache
python -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"
python tests/test_fold_fused.py --lengths 4096,8192,16384,32768 --reps 5
