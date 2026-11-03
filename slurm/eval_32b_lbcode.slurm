#!/bin/bash
#SBATCH --job-name=pf_e32lb
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# LongBench LCC / RepoBench-P for 1.7B -> 32B, third attempt. Raw prompts: Qwen3-32B answers with prose ("Okay, let's
# see..."); chat prompts (results/e32_longbench_chat): many configs explain instead of writing code, at config-dependent
# rates. Here the assistant turn is pre-filled with a code fence "```\n", so every config must continue with code; the
# prediction is the first code line. -> results/e32_longbench_fence
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
export PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Qwen3-32B
python scripts/eval_longbench.py --projector $PREFILL_ROOT/checkpoints/s17to32n_mix8k/projector.pt \
  --out_dir results/e32_longbench_fence --tasks lcc,repobench-p --code_chat --code_prefix $'```\n'
