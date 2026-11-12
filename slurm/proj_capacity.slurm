#!/bin/bash
#SBATCH --job-name=pf_pcap
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-pcap/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Projector precision, branch proj-capacity (worktree prefillLLM-pcap), one GPU, everything sequential:
#   0. diagnostics, no training (scripts/diag_projector.py): layer-group / K-only / V-only oracles + token-level error,
#      Qwen3-0.6B -> 8B (r2_mix8k) and Llama-3.2-1B -> 3.1-8B (ll1to8_mix8k)
#   1. capacity that is free at inference (prefill/projector.ProjectorWide), continued from r2_mix8k, 6000 steps each,
#      same data order / LR for all three: ctrl (original structure) | w2 (full per-layer W2) | w2w1 (+ per-layer W1)
#      each evaluated with hyb and attn_small:0.1:sent on KL / RULER / QA; r2_mix8k itself evaluated the same way
# START_AT=<stage> resumes (stages below in order). Re-run after the OOM: START_AT=probe (ctrl results kept).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-pcap || exit 1
echo "worktree $(pwd) branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
START_AT=${START_AT:-smoke}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$PREFILL_ROOT/checkpoints
BASE=$CK/r2_mix8k/projector.pt
SM=$CK/pcap_smoke
FAILED=""
ev() { "$@" || { FAILED="$FAILED $*;"; echo "!!!!!!!! FAILED: $*"; }; }
COMMON="--init $BASE --max_steps 6000 --accum 4 --lr 2e-5 --warmup 200 --curriculum_frac 0 --val_every 1000 --val_per_type 10 \
  --save_every 500 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --seed 1"
EVAL="--configs hyb,attn_small:0.1:sent --tasks kl,ruler,qa"

if should smoke; then
  set -e
  step "smoke: CPU equivalence test, GPU fold check on real weights, tiny diag, 4 training steps of w2w1 + tiny eval"
  rm -rf $SM
  python tests/test_projector_wide.py
  python tests/check_wide_gpu.py
  python scripts/diag_projector.py --out_dir $SM/diag --configs hyb,or:0-8:kv,or:all:k --tasks kl,tokerr,ruler,qa \
    --n_kl 1 --n_tok_ruler 1 --n_tok_qa 1 --n_ruler 1 --n_qa 1
  grep -A6 "KL" $SM/diag/summary.md | head -8
  python scripts/train_stage2.py $COMMON --widen w2w1 --max_steps 4 --accum 2 --val_every 2 --val_per_type 1 --save_every 2 --out_dir $SM/w2w1
  python scripts/diag_projector.py --projector $SM/w2w1/projector.pt --out_dir $SM/w2w1_eval $EVAL --n_kl 1 --n_ruler 1 --n_qa 1
  set +e
fi
if should diag06; then
  step "diag: Qwen3-0.6B -> Qwen3-8B (r2_mix8k)"
  ev python scripts/diag_projector.py --projector $BASE --out_dir results/pdiag_06
fi
if should diagll; then
  step "diag: Llama-3.2-1B -> Llama-3.1-8B (ll1to8_mix8k)"
  ( source slurm/llama_env.sh && python scripts/diag_projector.py --projector $CK/ll1to8_mix8k/projector.pt --out_dir results/pdiag_ll ) \
    || { FAILED="$FAILED diagll;"; echo "!!!!!!!! FAILED: diagll"; }
fi
if should eval_base; then
  step "eval r2_mix8k as is (reference for the continued runs)"
  ev python scripts/diag_projector.py --projector $BASE --out_dir results/pcap_base $EVAL
fi
if should probe; then
  # job 28149423: train_w2 / train_w2w1 OOM'd on the L40S within 30 steps (ctrl ran fine): +218M / +292M fp32 params with
  # grads + AdamW states, and the wide forward kept an fp32 [36, T, 2048] activation. Fixed by --proj_ckpt (projector
  # recomputed in backward; identical numbers) + bf16 activations; this stage fails the job in minutes if it still OOMs.
  step "memory probe: w2w1 (largest) with --proj_ckpt on the longest sequences"
  set -e
  python scripts/train_stage2.py $COMMON --widen w2w1 --proj_ckpt --probe_longest --max_steps 3 --accum 1 --val_every 1000 \
    --save_every 1000 --out_dir $CK/pcap_probe
  grep -h peak_gib $CK/pcap_probe/train_log.jsonl | tail -1
  set +e
fi
for V in ctrl w2 w2w1; do
  if [[ -f results/pcap_$V/summary.md && $START_AT != train_$V && $START_AT != eval_$V ]]; then
    should train_$V; echo "[skip] $V: results/pcap_$V/summary.md exists"; continue
  fi
  if should train_$V; then
    step "train $V: 6000 more steps from r2_mix8k"
    W=""; [[ $V != ctrl ]] && W="--widen $V --proj_ckpt"
    ev python scripts/train_stage2.py $COMMON $W --out_dir $CK/pcap_$V
  fi
  if should eval_$V; then
    step "eval $V"
    ev python scripts/diag_projector.py --projector $CK/pcap_$V/projector.pt --out_dir results/pcap_$V $EVAL
  fi
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
