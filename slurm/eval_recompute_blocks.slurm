#!/bin/bash
#SBATCH --job-name=pf_rc_blocks
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Selective recompute, selection unit: single tokens vs fixed blocks (8/16/32/64 tokens) vs sentence/line segments,
# at 5% and 10% of the middle tokens (8B-attention scorer; 0.6B-attention with sentences as the cheap variant).
# Same examples as results/recompute. Quality only (L40S); TTFT of these configs is measured by eval_recompute_speed.slurm on H100. Results: results/recompute_blocks/. Restart skips finished tasks.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
python scripts/eval_recompute.py --out_dir results/recompute_blocks --configs big,hyb,attn_big:0.05,attn_big:0.1,attn_big:0.05:b8,attn_big:0.05:b16,attn_big:0.05:b32,attn_big:0.05:b64,attn_big:0.05:sent,attn_big:0.1:b8,attn_big:0.1:b16,attn_big:0.1:b32,attn_big:0.1:b64,attn_big:0.1:sent,attn_small:0.05:sent,attn_small:0.1:sent --tasks niah,ruler,qa
