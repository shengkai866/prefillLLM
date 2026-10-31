#!/bin/bash
#SBATCH --job-name=pf_mq20
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Multi-question reuse, round 3: 20 questions per document (results/multiq2_* had 5). Does "recompute a few % of NEW
# sentences per question, keep them" (ours-acc) stay ahead of the 8B prefix cache as the questions pile up, and how does
# its quality evolve as its cache fills? New pool multiq_eval20: 30 document sets per length (~3.5k / ~7.5k tokens),
# 10 gold SQuAD paragraphs x 2 questions, shuffled. F1 per question-index bin (1 / 2-5 / 6-10 / 11-20); speed = document
# prefill + a sequence of 20 different questions (totals after 1 / 5 / 10 / 15 / 20 questions), 4k-32k.
# Projectors: 0.6B r2_mix8k, 1.7B r5_s17_normB (-> 8B). Each model is isolated; restarting skips finished configs.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
CFG=big,hyb,ours:0.1:sent,ours-acc:0.01:sent,ours-acc:0.02:sent,ours-acc:0.03:sent,ours-acc:0.05:sent,spec-each:0.1

step "pool multiq_eval20 (CPU)"
python scripts/prepare_multiq.py --n_gold 10 --n_q 20 --name multiq_eval20 || exit 1

run() {  # $1 tag, $2 small model dir, $3 projector
  set -e
  step "$2 -> 8B: shared document, 20 questions"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_multiq.py --projector $CK/$3/projector.pt --out_dir results/multiq20_$1 \
    --pool multiq_eval20 --tasks qa,speed --configs $CFG --speed_configs $CFG --speed_lengths 4096,8192,16384,32768 --speed_nq 20
}
( run 06 Qwen3-0.6B r2_mix8k ); [ $? -eq 0 ] || { FAILED="$FAILED 0.6B"; step "0.6B FAILED, continuing"; }
( run 17 Qwen3-1.7B r5_s17_normB ); [ $? -eq 0 ] || FAILED="$FAILED 1.7B"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
