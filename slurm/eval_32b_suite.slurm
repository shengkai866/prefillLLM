#!/bin/bash
#SBATCH --job-name=pf_e32
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# The current benchmark (REPORT_2026-10-28_followup.md) for Qwen3-1.7B -> Qwen3-32B with the round-6 in_norm projector
# (slurm/run_r6_32b_norm.slurm -> checkpoints/s17to32n_mix8k). One H100: the 32B (65.5 GB bf16) does not fit an L40S.
# Submitted with --dependency=afterok on the training job.
#   A main     : NIAH, RULER 4k/8k, QA 4k/8k, GSM8K 8-shot, code; 32B / 1.7B alone, hyb, ours 10/20%, drop-rest 20%,
#                SpecPrefill 10/20/30%                                         -> results/e32_main
#   B speed    : TTFT, every method CUDA-graphed, 4k/8k/16k (32k does not fit one H100 next to the 32B) -> results/e32_speed
#   C long     : QA + NIAH at 16k (32k bucket skipped, same reason)               -> results/e32_long
#   D multiq   : shared document, 5 questions                                    -> results/e32_multiq
#   E longbench: gov_report / QMSum / LCC / RepoBench-P + 32-shot GSM8K           -> results/e32_longbench
#   F rarecache: fit the 1.7B -> 32B maps, then rarecache-s 0/10/30/50%           -> results/e32_rare
# Each part is isolated (a failure does not stop the others) and restarting skips finished tasks.
# Run only some parts: sbatch --export=ALL,PARTS="C E" slurm/eval_32b_suite.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B
export PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Qwen3-32B
P=$CK/s17to32n_mix8k/projector.pt
PARTS=${PARTS:-"A B C D E F"}
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

CFG=big,small,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
R=rarecache-s:0,rarecache-s:0.1,rarecache-s:0.3,rarecache-s:0.5
LENS=4096,8192,16384

part_A() {
  set -e
  step "A main: NIAH / RULER / QA / GSM8K / code"
  python scripts/eval_recompute.py --projector $P --out_dir results/e32_main --tasks niah,ruler,qa,gsm8k,code --configs $CFG
}
part_B() {
  set -e
  step "B speed: every method CUDA-graphed"
  python scripts/eval_recompute.py --projector $P --out_dir results/e32_speed --tasks speed_allgraph --configs hyb \
    --speed_configs $CFG --speed_lengths $LENS
}
part_C() {
  set -e
  step "C long: QA + NIAH at 16k"
  python scripts/eval_recompute.py --projector $P --out_dir results/e32_long --tasks niah_long,qa_long --long_max 16384 \
    --configs big,small,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.3
}
part_D() {
  set -e
  step "D multiq: shared document, several questions"
  python scripts/eval_multiq.py --projector $P --out_dir results/e32_multiq --tasks qa,speed --speed_lengths $LENS
}
part_E() {
  set -e
  step "E LongBench + 32-shot GSM8K"
  python scripts/eval_longbench.py --projector $P --out_dir results/e32_longbench
}
part_F() {
  set -e
  F=$CK/rarecache_1.7B_to_32B/fit.pt
  step "F RaReCache: fit 1.7B -> 32B maps"
  [ -f $F ] || python scripts/fit_rarecache.py --out $F
  step "F RaReCache (sink/tail forced): speed, NIAH, RULER, QA, GSM8K, code"
  python scripts/eval_recompute.py --projector $P --rarecache_fit $F --out_dir results/e32_rare \
    --tasks speed,niah,ruler,qa,gsm8k,code --configs $R \
    --speed_configs big,hyb,attn_small:0.1:sent,specprefill:0.1,specprefill:0.3,$R --speed_graph --speed_lengths $LENS
}

[ -f $P ] || { echo "missing projector $P"; exit 1; }
# Run each part as a plain statement: inside "( f ) || ..." bash would silently ignore the set -e in f.
for x in $PARTS; do
  ( part_$x ); [ $? -eq 0 ] || { FAILED="$FAILED $x"; step "part $x FAILED, continuing"; }
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
