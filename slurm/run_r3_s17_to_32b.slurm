#!/bin/bash
#SBATCH --job-name=pf_r3_32b
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Round 3b: Qwen3-1.7B prefills for Qwen3-32B, same recipe as the round-2 mix-8k run.
# Training splits the 32B (65.5 GB bf16) over 2 H100s (PREFILL_LARGE_MAXMEM); evaluation fits on one H100.
# Stage-2 data: the round-2 self-distillation stores (responses written by the 8B, teacher logits come from the 32B,
# so the distillation is off-policy for the 32B; regenerating with the 32B would take ~2 days).
#   smoke   : 6 stage-1 steps + 4 stage-2 steps (8k prompts) + a tiny single-GPU eval, to fail fast on memory/devices
#   stage1  : 8000 steps            -> $PREFILL_ROOT/checkpoints/s17to32_stage1
#   stage2  : 12000 steps           -> $PREFILL_ROOT/checkpoints/s17to32_mix8k
#   eval_rc : RULER / QA / NIAH with and without selective recompute + TTFT -> results/r3_s17_to_32b/
#   eval_v2 : KL per data type, code next-line, GSM8K                       -> results/r3_s17_to_32b_v2/
# Restart from a step:  sbatch --export=ALL,START_AT=stage2 slurm/run_r3_s17_to_32b.slurm
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
export PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Qwen3-32B
SPLIT="24GiB,76GiB"   # 32B layers: ~24 GB on GPU 0 (which also holds the 1.7B, projector and losses), rest on GPU 1
START_AT=${START_AT:-smoke}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
CK=$PREFILL_ROOT/checkpoints
SM=$CK/s17to32_smoke
S1=$CK/s17to32_stage1
S2=$CK/s17to32_mix8k

if should smoke; then
  step "smoke: 2-GPU training steps + single-GPU eval"
  rm -rf $SM
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage1.py --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/s2 --max_steps 4 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  python scripts/eval_recompute.py --projector $SM/s2/projector.pt --out_dir $SM/eval --tasks niah,speed \
    --configs hyb,attn_small:0.1:sent --speed_configs big,hyb,attn_small:0.1:sent --speed_lengths 16384 --speed_reps 1
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
fi
if should stage1; then
  step "stage1: Qwen3-1.7B -> Qwen3-32B representation alignment"
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage1.py --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 --resume $S1/projector.pt
fi
if should stage2; then
  step "stage2: mixed data, prompts <= 8k (round-2 mix8k recipe)"
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2
fi
if should eval_rc; then
  step "eval: selective recompute (RULER / QA / NIAH / TTFT), single GPU"
  python scripts/eval_recompute.py --projector $S2/projector.pt --out_dir results/r3_s17_to_32b \
    --configs big,small,hyb,attn_big:0.05,attn_big:0.1:sent,attn_small:0.05:sent,attn_small:0.1:sent \
    --speed_configs big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05 --speed_lengths 4096,8192,16384
fi
if should eval_v2; then
  step "eval: KL per data type, code, GSM8K, single GPU"
  python scripts/evaluate_v2.py --projectors s17to32=$S2/projector.pt --curve_runs "" --tasks kl,code,gsm8k --out_dir results/r3_s17_to_32b_v2
fi
step "all done"
