#!/bin/bash
#SBATCH --job-name=pf_rcctl
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch moe-router: benchmark of the control rcctl_06 (0.6B->8B, same 4000-step fine-tune as rc_06 but WITHOUT recompute)
# -> results/rcctl_06_main. Question: does fine-tuning alone (no recompute) also lower QA 4k F1? Compare with
# main's results/fused_06_main (base) and results/rc_06_main (rc).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-0.6B
python scripts/eval_recompute.py --projector $PREFILL_ROOT/checkpoints/rcctl_06/projector.pt --out_dir results/rcctl_06_main \
  --tasks niah,ruler,qa,gsm8k,code --configs hyb,attn_small:0.1:sent
