#!/bin/bash
#SBATCH --job-name=pf_r3_s17
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Round 3a: Qwen3-1.7B (instead of 0.6B) prefills for Qwen3-8B. Same recipe as the round-2 mix-8k run so the only
# change is the small model: stage 1 (8000 steps, web text) -> stage 2 (12000 steps, round-2 self-distillation data
# sd_mix4k + sd_mixlong reused as is: same tokenizer, responses come from the 8B) -> evaluation:
#   eval_rc : RULER / QA / NIAH with and without selective recompute + TTFT  -> results/r3_small17/
#   eval_v2 : KL per data type, code next-line, GSM8K                        -> results/r3_small17_v2/
# Restart from a step:  sbatch --export=ALL,START_AT=stage2 slurm/run_r3_small17.slurm   (stage1 stage2 eval_rc eval_v2)
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
START_AT=${START_AT:-stage1}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
S1=$PREFILL_ROOT/checkpoints/s17_stage1
S2=$PREFILL_ROOT/checkpoints/s17_mix8k

if should stage1; then
  step "stage1: Qwen3-1.7B -> Qwen3-8B representation alignment"
  python scripts/train_stage1.py --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 --resume $S1/projector.pt
fi
if should stage2; then
  step "stage2: mixed data, prompts <= 8k (round-2 mix8k recipe)"
  python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 --val_every 500 \
    --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2
fi
if should eval_rc; then
  step "eval: selective recompute (RULER / QA / NIAH / TTFT)"
  python scripts/eval_recompute.py --projector $S2/projector.pt --out_dir results/r3_small17 \
    --configs big,small,hyb,random:0.1,attn_big:0.05,attn_big:0.1,attn_big:0.1:sent,attn_small:0.05:sent,attn_small:0.1:sent \
    --speed_configs big,small,hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05,attn_big:0.1:sent
fi
if should eval_v2; then
  step "eval: KL per data type, code, GSM8K"
  python scripts/evaluate_v2.py --projectors s17mix8k=$S2/projector.pt --curve_runs "" --tasks kl,code,gsm8k --out_dir results/r3_small17_v2
fi
step "all done"
