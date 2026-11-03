#!/bin/bash
#SBATCH --job-name=pf_e32x
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Two follow-ups of slurm/eval_32b_suite.slurm (1.7B -> 32B, round-6 in_norm projector s17to32n_mix8k), one H100:
#   A multiq20 : shared document, 20 questions (pool multiq_eval20), incl. the per-question 1 / 2 / 3% cumulative
#                configs that the suite did not have; speed = document + a sequence of 20 questions, 4k-16k
#                -> results/e32_multiq20
#   B code_chat: LongBench LCC / RepoBench-P through the chat template (thinking off), first code line of the answer:
#                Qwen3-32B answers the raw code prompts with prose, which made results/e32_longbench's code columns invalid
#                -> results/e32_longbench_chat
# Each part is isolated; restarting skips finished configs / tasks.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
export PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Qwen3-32B
P=$PREFILL_ROOT/checkpoints/s17to32n_mix8k/projector.pt
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
MQ=big,hyb,ours:0.1:sent,ours-acc:0.01:sent,ours-acc:0.02:sent,ours-acc:0.03:sent,spec-each:0.1

part_A() {
  set -e
  step "A multiq, 20 questions"
  python scripts/eval_multiq.py --projector $P --out_dir results/e32_multiq20 --pool multiq_eval20 --tasks qa,speed \
    --configs $MQ --speed_configs $MQ --speed_lengths 4096,8192,16384 --speed_nq 20
}
part_B() {
  set -e
  step "B LongBench code through the chat template"
  python scripts/eval_longbench.py --projector $P --out_dir results/e32_longbench_chat --tasks lcc,repobench-p --code_chat
}
( part_A ); [ $? -eq 0 ] || { FAILED="$FAILED A"; step "part A FAILED, continuing"; }
( part_B ); [ $? -eq 0 ] || FAILED="$FAILED B"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
