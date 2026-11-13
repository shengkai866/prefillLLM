#!/bin/bash
#SBATCH --job-name=pf_moe1826
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moelayers/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Layer-routed MoE projector (prefill/projector.ProjectorMoE), branch moe-layers (worktree prefillLLM-moelayers), one L40S:
# Qwen3-0.6B -> Qwen3-8B with TWO independent projectors, one for large layers $SPANS (default 18-26, the layer group
# whose true K/V recovers most of the gap to the 8B, results/pdiag_06), one for the other layers. Trained from scratch
# with the recipe of the single projector r2_mix8k (the baseline): stage 1 = representation alignment (8000 steps,
# lr 2e-4), stage 2 = KL to the 8B's own answers on sd_mix4k + sd_mixlong, prompts <= 8k (12000 steps, lr 5e-5).
#   smoke   CPU test, GPU fold check on real weights, a few steps of both stages (+ memory probe on the longest prompts), tiny eval
#   stage1 / stage2
#   eval    scripts/diag_projector.py: KL / RULER 4k,8k / QA for hyb and attn_small:0.1:sent + true-K/V oracle of the
#           span + token-level K/V error per layer group   -> results/${TAG}_eval   (baseline: results/pcap_base, pdiag_06)
#   main    scripts/eval_recompute.py: NIAH / RULER / QA / GSM8K / code -> results/${TAG}_main (baseline: main's results/fused_06_main)
# Restart: sbatch --export=ALL,START_AT=<stage> ... ; STOP_AFTER=<stage> ends after that stage; SPANS / TAG select another split.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moelayers || exit 1
echo "worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
SPANS=${SPANS:-18-26}
TAG=${TAG:-moe${SPANS//[-,]/}}
START_AT=${START_AT:-smoke}
STOP_AFTER=${STOP_AFTER:-}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ -n $stopped ]]; then return 1; fi; if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; [[ $1 == "$STOP_AFTER" ]] && stop_next=1; return 0; fi; return 1; }
after() { [[ -n $stop_next ]] && stopped=1; return 0; }
CK=$PREFILL_ROOT/checkpoints
SM=$CK/${TAG}_smoke
S1=$CK/${TAG}_stage1
S2=$CK/${TAG}_mix8k
P=$S2/projector.pt
FAILED=""
ev() { "$@" || { FAILED="$FAILED $*;"; echo "!!!!!!!! FAILED: $*"; }; }
S2ARGS="--sd sd_mix4k,sd_mixlong --max_prompt 8064"
echo "SPANS=$SPANS TAG=$TAG START_AT=$START_AT STOP_AFTER=${STOP_AFTER:-none}"

if should smoke; then
  set -e
  step "smoke: CPU test, GPU fold check, a few training steps of both stages, memory probe, tiny eval"
  rm -rf $SM
  python tests/test_projector_moe.py
  python tests/smoke_moe.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert"
  python tests/check_moe_gpu.py --spans $SPANS
  python scripts/train_stage1.py --moe_spans $SPANS --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/s2 --max_steps 4 --accum 2 --val_every 2 --val_per_type 1 \
    --save_every 1000 --save_step_every 0 --curriculum_frac 0 $S2ARGS
  python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/probe --max_steps 3 --accum 1 --val_every 1000 \
    --save_every 1000 --save_step_every 0 --probe_longest $S2ARGS
  grep -h peak_gib $SM/probe/train_log.jsonl | tail -1
  python scripts/diag_projector.py --projector $SM/s2/projector.pt --out_dir $SM/eval --configs hyb,attn_small:0.1:sent,or:$SPANS:kv \
    --tasks kl,tokerr,ruler,qa --n_kl 1 --n_tok_ruler 1 --n_tok_qa 1 --n_ruler 1 --n_qa 1
  python scripts/eval_recompute.py --projector $SM/s2/projector.pt --out_dir $SM/main --tasks niah,gsm8k,code \
    --configs hyb,attn_small:0.1:sent --n_gsm8k 2 --n_code 2
  set +e
  after
fi
if should stage1; then
  step "stage1: representation alignment, ProjectorMoE spans $SPANS"
  python scripts/train_stage1.py --moe_spans $SPANS --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 \
    --resume $S1/projector.pt || exit 1
  after
fi
if should stage2; then
  step "stage2: mixed data, prompts <= 8k (round-2 mix8k recipe)"
  python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 10 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    $S2ARGS --out_dir $S2 || exit 1
  after
fi
if should eval; then
  step "eval: KL / RULER / QA (hyb, 10 % recompute, true K/V in the span) + token-level K/V error"
  ev python scripts/diag_projector.py --projector $P --out_dir results/${TAG}_eval \
    --configs hyb,attn_small:0.1:sent,or:$SPANS:kv --tasks kl,tokerr,ruler,qa
  after
fi
if should main; then
  step "eval main: NIAH / RULER / QA / GSM8K / code"
  ev python scripts/eval_recompute.py --projector $P --out_dir results/${TAG}_main --tasks niah,ruler,qa,gsm8k,code \
    --configs hyb,attn_small:0.1:sent,attn_small:0.2:sent
  after
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
