#!/bin/bash
#SBATCH --job-name=pf_mid4b
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Iso-latency baseline: a mid-size model of the target's family ALONE (default Qwen3-4B for the Qwen3-8B targets) on
# every benchmark the pairs were evaluated on, so "ours (small -> 8B)" can be compared with "just run a smaller model".
# The mid model is loaded as the "small" model next to Qwen3-8B and run with the existing `small` config (its own prefill
# and decoding; the projector is loaded but never used by `small`). Only `small` is evaluated: 8B / ours / SpecPrefill
# come from the pair runs (results/fused_<tag>_*, quality_<pair>, bench_<pair>). The speed step also times `big` in
# the same run as the reference for speedups.
# Outputs results/mid_<NAME>_<step> (NAME = model dir without "Qwen3-", e.g. mid_4B_main):
#   speed     TTFT, CUDA graph, 4k-32k (big, small)            -> mid_4B_speed
#   main      NIAH / RULER / QA / GSM8K / code                  -> mid_4B_main
#   long      16k / 32k NIAH and QA                             -> mid_4B_long
#   multiq    shared document, 5 questions (+ per-question time) -> mid_4B_multiq
#   longbench gov_report / QMSum / LCC / RepoBench-P / GSM8K 32-shot -> mid_4B_longbench
#   quality   QuALITY single question + multi-question          -> mid_4B_quality, mid_4B_quality_mq
#   bench     long-context suite (8 tasks, 64k cap, YaRN x3 for the long ones) -> mid_4B_bench
#   summary   scripts/summarize_mid.py -> results/mid_summary.md
#   sbatch slurm/eval_mid.slurm
#   sbatch --export=ALL,MID=Qwen3-4B,STEPS="main long" slurm/eval_mid.slurm      # some steps only
# Every step runs with --force and is isolated (a failure is reported at the end and does not stop the rest).
set -o pipefail
CODE=${CODE:-$SLURM_SUBMIT_DIR}  # checkout holding scripts/summarize_mid.py (env.sh cds to the main checkout for the evals)
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
M=$PREFILL_ROOT/models
MID=${MID:-Qwen3-4B}
export PREFILL_SMALL_DIR=$M/$MID   # large stays Qwen3-8B (default)
[ -f $PREFILL_SMALL_DIR/config.json ] || { echo "missing $PREFILL_SMALL_DIR (python scripts/download_models.py Qwen/$MID)"; exit 1; }
P=$PREFILL_ROOT/checkpoints/r2_mix8k/projector.pt  # loaded by the Engine, unused by `small`
NAME=${MID#Qwen3-}
R=results/mid_$NAME
LEN=4096,8192,16384,32768
echo "MID=$MID small=$PREFILL_SMALL_DIR large=${PREFILL_LARGE_DIR:-default} out=${R}_<step> code=$CODE"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $S"; echo "!!!!!!!! FAILED: $*"; }; }

for S in ${STEPS:-speed main long multiq longbench quality bench summary}; do
  step "$MID / $S"
  case $S in
    speed)     ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph --configs small \
                 --speed_configs big,small --speed_lengths $LEN --force ;;
    main)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs small --force ;;
    long)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_long --tasks niah_long,qa_long --configs small --force ;;
    multiq)    ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_multiq --tasks qa,speed --configs small --speed_configs big,small --force ;;
    longbench) ev python scripts/eval_longbench.py --projector $P --out_dir ${R}_longbench --configs small --force ;;
    quality)   ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_quality --tasks quality --configs small --force
               ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_quality_mq --pool quality_multiq --tasks qa --configs small --q_max 320 --force ;;
    bench)     ev python scripts/eval_bench.py --projector $P --out_dir ${R}_bench --max_len 65536 --tasks lastingbench,musique,qasper,longhealth,contractnli --configs small --force
               ev python scripts/eval_bench.py --projector $P --out_dir ${R}_bench --max_len 65536 --tasks loogle,loogle_long,clipper --configs small --force ;;
    summary)   if [ -f $CODE/scripts/summarize_mid.py ]; then ev python $CODE/scripts/summarize_mid.py --results $PWD/results --mid $NAME
               else echo "no $CODE/scripts/summarize_mid.py; run it by hand"; fi ;;
    *) echo "unknown step $S"; FAILED="$FAILED $S" ;;
  esac
done
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
