#!/bin/bash
#SBATCH --job-name=pf_sparse
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Sparse-attention prefill baseline: FlexPrefill (ICLR 2025; official Triton op, prefill/sparse.py), everything in ONE
# job on ONE L40S. The large model prefills the whole prompt itself with block-sparse attention (official defaults:
# gamma 0.9, tau 0.1, min_budget 512, block 128); gamma 0.95 as a second operating point where the prompt length is
# fixed (main / long / QuALITY, for the Pareto plots). Its per-layer block selection runs on the host, so TTFT is eager.
# Targets (TARGETS, default both 8B targets; the 32B target needs an H100 and is not here):
#   q8   Qwen3-8B         (compared with 0.6B / 1.7B / Llama-1B -> Qwen3-8B)   -> results/sparse_q8_<step>
#   ll8  Llama-3.1-8B-Ins (compared with Llama-1B / Qwen-1.7B -> Llama-3.1-8B) -> results/sparse_ll8_<step>
# Steps (STEPS): smoke (tests/gpu_check_flexprefill.py; a failure skips the target), speed (TTFT 4k-32k, 8B dense in the
# same run as reference), main (NIAH / RULER / QA / GSM8K / code), long (16k / 32k), quality (QuALITY single + multi-
# question), multiq (5 questions per document: FlexPrefill prefills the document once, questions reuse the cache),
# longbench, e2e (q8 only: end-to-end latency vs output length + memory, scripts/bench_e2e.py -> results/e2e_flex_q8),
# bench (long-context suite, 64k cap, YaRN x3 above 40k for Qwen3). Then the summaries: summarize_mid, paired_ci,
# plot_pareto, summarize_e2e (code in $CODE, the checkout the job was submitted from).
#   sbatch slurm/sparse_baseline.slurm
#   sbatch --export=ALL,TARGETS=q8,STEPS="main quality" slurm/sparse_baseline.slurm
set -o pipefail
CODE=${CODE:-$SLURM_SUBMIT_DIR}
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd $CODE
export PYTHONPATH=$CODE
CK=$PREFILL_ROOT/checkpoints
OUT=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/results
LEN=4096,8192,16384,32768
F1=flexprefill
FX=flexprefill,flexprefill:0.95
echo "code=$CODE out=$OUT"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

run_target() {  # in a subshell: the target's environment must not leak
  ( T=$1
    case $T in
      q8)  P=$CK/r2_mix8k/projector.pt ;;  # the Engine needs a projector; FlexPrefill never uses it
      ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt ;;
      *) echo "unknown target $T"; echo "FAILED_TARGET: $T"; exit 1 ;;
    esac
    R=$OUT/sparse_$T
    F=""
    ev() { "$@" || { F="$F $T:$S"; echo "!!!!!!!! FAILED: $*"; }; }
    echo "target $T: large=${PREFILL_LARGE_DIR:-default} small=${PREFILL_SMALL_DIR:-default} data=${PREFILL_DATA_DIR:-default}"
    for S in ${STEPS:-smoke speed main long quality multiq longbench e2e bench}; do
      step "$T / $S"
      case $S in
        smoke)     python tests/gpu_check_flexprefill.py || { echo "FAILED_TARGET: $T:smoke (FlexPrefill does not run correctly here)"; exit 1; } ;;
        speed)     ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph --configs big \
                     --speed_configs big,$FX --speed_lengths $LEN --force ;;
        main)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs $FX --force ;;
        long)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_long --tasks niah_long,qa_long --configs $FX --force ;;
        quality)   ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_quality --tasks quality --configs $FX --force
                   ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_quality_mq --pool quality_multiq --tasks qa --configs flex --q_max 320 --force ;;
        multiq)    ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_multiq --tasks qa,speed --configs flex --speed_configs big,flex --force ;;
        longbench) ev python scripts/eval_longbench.py --projector $P --out_dir ${R}_longbench --configs $F1 --force ;;
        e2e)       [ $T = q8 ] && ev python scripts/bench_e2e.py --projector $P --out_dir $OUT/e2e_flex_q8 --tasks decode,memory --configs big,$F1 ;;
        bench)     ev python scripts/eval_bench.py --projector $P --out_dir ${R}_bench --max_len 65536 --tasks lastingbench,musique,qasper,longhealth,contractnli --configs $F1 --force
                   ev python scripts/eval_bench.py --projector $P --out_dir ${R}_bench --max_len 65536 --tasks loogle,loogle_long,clipper --configs $F1 --force ;;
        *) echo "unknown step $S"; F="$F $T:$S" ;;
      esac
    done
    [ -z "$F" ] || echo "FAILED_TARGET:$F" )
}

for T in ${TARGETS:-q8 ll8}; do
  out=$(run_target $T 2>&1 | tee >(cat >&2) | grep "^FAILED_TARGET:" | tail -1)
  [ -z "$out" ] || FAILED="$FAILED ${out#FAILED_TARGET:}"
done
step "summaries"
python scripts/summarize_mid.py --results $OUT || FAILED="$FAILED summarize_mid"
python scripts/paired_ci.py --results $OUT || FAILED="$FAILED paired_ci"
python scripts/plot_pareto.py --results $OUT || FAILED="$FAILED plot_pareto"
python scripts/summarize_e2e.py --results $OUT || FAILED="$FAILED summarize_e2e"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
