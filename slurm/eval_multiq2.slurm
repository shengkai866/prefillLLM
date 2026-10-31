#!/bin/bash
#SBATCH --job-name=pf_mq2
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
# Multi-question reuse, round 2: can the per-question cost drop to the 8B prefix cache's ~40 ms while keeping the
# quality of "projected cache + recompute 10% per question" (REPORT_2026-10-28_followup.md sec 3.3 / 6.2)?
#   ours-acc:f      recompute f of NEW sentences per question, rows persist (f = 10 / 5 / 3%)
#   pre:f[:..:g]    question-agnostic recompute of f at DOCUMENT time (small-model attention of the last document
#                   tokens), then nothing (g = 0) or g of new sentences per question (cumulative)
#   + baselines big (8B prefix cache), hyb, ours (10% per question, not kept), spec-each 10%
# Same multiq_eval pool as results/multiq_06|17 (30 doc sets x 5 SQuAD questions, ~3.5k / 7.5k tokens); speed with
# random tokens at 4k-32k. Projectors: 0.6B r2_mix8k, 1.7B r5_s17_normB (-> 8B). Each model is isolated.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
CFG=big,hyb,ours:0.1:sent,ours-acc:0.1:sent,ours-acc:0.05:sent,ours-acc:0.03:sent,pre:0.1:sent,pre:0.2:sent,pre:0.1:sent:0.03,pre:0.05:sent:0.03,spec-each:0.1

step "smoke test (CPU, tiny models)"
(cd tests && python smoke_multiq.py 2>&1 | grep -E "pre |Error|assert" | tail -4)

run() {  # $1 tag, $2 small model dir, $3 projector
  set -e
  step "$2 -> 8B: shared document, several questions"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_multiq.py --projector $CK/$3/projector.pt --out_dir results/multiq2_$1 \
    --tasks qa,speed --configs $CFG --speed_configs $CFG --speed_lengths 4096,8192,16384,32768
}
( run 06 Qwen3-0.6B r2_mix8k ); [ $? -eq 0 ] || { FAILED="$FAILED 0.6B"; step "0.6B FAILED, continuing"; }
( run 17 Qwen3-1.7B r5_s17_normB ); [ $? -eq 0 ] || FAILED="$FAILED 1.7B"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
