#!/bin/bash
#SBATCH --job-name=pf4_eval
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Evaluation: big alone vs small alone vs hybrid (stage1 and stage2 projectors): speed, ppl/KL, oracle, NIAH, GSM8K.
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/evaluate.py --tasks speed,ppl,oracle,niah,gsm8k --out_dir results/main
echo "== summary =="; cat results/main/summary.md
