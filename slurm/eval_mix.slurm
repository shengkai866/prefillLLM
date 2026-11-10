#!/bin/bash
#SBATCH --job-name=pf_mix
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-pdrop/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch proj-drop: recompute + project + drop, 0.6B -> 8B (r2_mix8k). Recompute 10 % (small-model attention, sentences);
# of the remaining middle tokens keep a share with projected KV and drop the rest:
#   keep 1 (= ours) | 0.75 | 0.5 | 0.25 (least attended dropped) | 0 (= drop the rest) | random 0.5 (control)
# Test first (tests/test_mix.py), then eval_recompute niah / ruler / qa / gsm8k / code -> results/mix_06
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-pdrop/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-0.6B
P=$PREFILL_ROOT/checkpoints/r2_mix8k/projector.pt
python tests/test_mix.py $P || { echo "!!!!!!!! test FAILED"; exit 1; }
python scripts/eval_recompute.py --projector $P --out_dir results/mix_06 --tasks niah,ruler,qa,gsm8k,code \
  --configs attn_small:0.1:sent,attn_small:0.1k0.75:sent,attn_small:0.1k0.5:sent,attn_small:0.1k0.25:sent,drop-attn_small:0.1:sent,attn_small:0.1kr0.5:sent
