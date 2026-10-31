#!/bin/bash
#SBATCH --job-name=pf_llgen
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Llama pair, step 1 of 2: stage-2 self-distillation data, Llama-3.1-8B-Instruct answering / continuing the prompts of
# the Llama token pools (slurm/prep_llama.slurm), same settings as round 2 (slurm/run_round2.slurm): held-out mixtest
# (greedy), mix4k_train and mixlong_train (sampled). Two L40S, one generation process per GPU over interleaved shards,
# then a merge. ~9 h of H100 time for Qwen3-8B. Resumable (finished shards are kept). Step 2: slurm/run_llama.slurm.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
gen() {  # pool out [--greedy]
  step "generate $2 from $1"
  CUDA_VISIBLE_DEVICES=0 python scripts/gen_selfdistill.py --pool $1 --out $2 ${3:-} --worker 0 --n_workers 2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/gen_selfdistill.py --pool $1 --out $2 ${3:-} --worker 1 --n_workers 2 &
  wait -n && wait -n  # both workers must succeed
  CUDA_VISIBLE_DEVICES=0 python scripts/gen_selfdistill.py --pool $1 --out $2 ${3:-}  # merge
}
gen mixtest sd_mixtest --greedy
gen mix4k_train sd_mix4k
gen mixlong_train sd_mixlong
step "all done"
