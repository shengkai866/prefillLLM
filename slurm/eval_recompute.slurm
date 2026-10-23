#!/bin/bash
#SBATCH --job-name=pf_recompute
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Selective recompute (prefill/recompute.py) with the round-2 mix-8k projector: quality vs fraction of middle
# tokens recomputed by the 8B for several token scorers, plus TTFT. No training. H100 for comparable speed numbers.
# Results: results/recompute/{summary.md,results.json,generations.jsonl,figures/}. Restart skips finished tasks.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/eval_recompute.py --out_dir results/recompute
