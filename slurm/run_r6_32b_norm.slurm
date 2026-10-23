#!/bin/bash
#SBATCH --job-name=pf_r6_32n
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
# Round 6: Qwen3-1.7B -> Qwen3-32B again, from scratch, with RMS-normalised small-model inputs (--in_norm).
# Same recipe as round 3b (slurm/run_r3_s17_to_32b.slurm) except in_norm, i.e. the 1.7B->8B normA recipe that fixed
# 1.7B->8B in round 5 (RULER no-recompute 0.49/0.40 -> 0.84/0.74). Expected ~20 h (stage1 ~3.6 h, stage2 ~13 h, evals ~3.5 h).
#   smoke   : 6 stage-1 steps + 4 stage-2 steps (8k prompts) + a tiny single-GPU eval, to fail fast on memory/devices
#   stage1  : 8000 steps            -> $PREFILL_ROOT/checkpoints/s17to32n_stage1
#   stage2  : 12000 steps           -> $PREFILL_ROOT/checkpoints/s17to32n_mix8k
#   eval_rc : RULER / QA / NIAH with and without selective recompute -> results/r6_s17_to_32b_norm/
#             (32B alone / 1.7B alone: reuse results/r3_s17_to_32b, same examples and GPU type)
#   eval_v2 : KL per data type, code next-line, GSM8K, KL + retrieval at every checkpoint -> results/r6_s17_to_32b_norm_v2/
# Restart from a step:  sbatch --export=ALL,START_AT=stage2 slurm/run_r6_32b_norm.slurm
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
SM=$CK/s17to32n_smoke
S1=$CK/s17to32n_stage1
S2=$CK/s17to32n_mix8k
RES=results/r6_s17_to_32b_norm

if should smoke; then
  step "smoke: 2-GPU training steps (in_norm) + single-GPU eval"
  rm -rf $SM
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage1.py --in_norm --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/s2 --max_steps 4 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  python scripts/eval_recompute.py --projector $SM/s2/projector.pt --out_dir $SM/eval --tasks niah --configs hyb,attn_small:0.1:sent
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
fi
if should stage1; then
  step "stage1: Qwen3-1.7B -> Qwen3-32B representation alignment, in_norm"
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage1.py --in_norm --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 \
    --time_budget_hours 22 --resume $S1/projector.pt
fi
if should stage2; then
  step "stage2: mixed data, prompts <= 8k (round-2 mix8k recipe)"
  PREFILL_LARGE_MAXMEM=$SPLIT python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2
fi
if should eval_rc; then
  step "eval: RULER / QA / NIAH with and without selective recompute, single GPU"
  python scripts/eval_recompute.py --projector $S2/projector.pt --out_dir $RES --tasks niah,ruler,qa \
    --configs hyb,attn_small:0.05:sent,attn_small:0.1:sent,attn_big:0.05
fi
if should eval_v2; then
  step "eval: KL per data type, code, GSM8K; KL + retrieval at every checkpoint, single GPU"
  python scripts/evaluate_v2.py --projectors s17to32n=$S2/projector.pt --baselines "" --tasks kl,gsm8k,code,curve \
    --curve_runs s17to32n=$S2 --out_dir ${RES}_v2
fi
step "all done"
