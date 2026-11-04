#!/bin/bash
#SBATCH --job-name=pf_fused_all
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Every L40S experiment re-run with the fused projector -> KV kernel (PREFILL_FOLD=fused, prefill/fold_fused.py), on ONE
# L40S, in one job. Pairs (PAIRS, default all five that fit a 48 GB L40S; the 32B pair needs the H100 and is not here):
#   06      Qwen3-0.6B -> Qwen3-8B            r2_mix8k        results/fused_06_<step>
#   17      Qwen3-1.7B -> Qwen3-8B            r5_s17_normB    results/fused_17_<step>
#   ll      Llama-3.2-1B -> Llama-3.1-8B      ll1to8_mix8k    results/fused_ll1to8_<step>
#   xq17ll8 Qwen3-1.7B -> Llama-3.1-8B        q17toll8_mix8k  results/fused_q17toll8_<step>
#   xll1q8  Llama-3.2-1B -> Qwen3-8B          ll1toq8_mix8k   results/fused_ll1toq8_<step>
# Phase A (first ~1.5 h): speed_allgraph for every pair, fused (all methods) and PREFILL_FOLD=eager (ours only) -> TTFT of large /
#   small / SpecPrefill / ours with and without the fused kernel, summarised by scripts/summarize_speed.py -> results/speed_summary.md.
# Phase B, per pair: main (NIAH / RULER / QA / GSM8K / code), long (16k / 32k), multiq, LongBench + GSM8K 32-shot,
#   RaReCache (existing fit.pt reused), QuALITY (in place: results/quality_<pair>, quality_mq_<pair>), long-context bench
#   (in place: results/bench_<pair>, summarised by scripts/summarize_bench.py). Everything runs with --force (nothing is
#   skipped); every step is isolated (a failure is reported at the end and does not stop the rest).
# Estimated ~14 h per pair (~3 days total). Restart with START_PAIR=<pair> [START_STEP=<step>] to resume.
#   sbatch slurm/rerun_fused_all.slurm
#   sbatch --export=ALL,PAIRS="06 17" slurm/rerun_fused_all.slurm
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
export PREFILL_FOLD=fused
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
PAIRS=${PAIRS:-"06 17 ll xq17ll8 xll1q8"}
START_PAIR=${START_PAIR:-}
START_STEP=${START_STEP:-}
LEN=4096,8192,16384,32768
RC=rarecache-s:0,rarecache-s:0.1,rarecache-s:0.3,rarecache-s:0.5
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $PAIR:$STEP"; echo "!!!!!!!! FAILED: $*"; }; }

setup_pair() {  # sets the model dirs (in the caller's subshell), projector P, TAG, SMALL (same-tokenizer pairs only), F (RaReCache fit)
  PAIR=$1
  unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
  SMALL=small,
  case $PAIR in
    06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt; TAG=06; F=$CK/rarecache_0.6B/fit.pt ;;
    17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt; TAG=17; F=$CK/rarecache_1.7B/fit.pt ;;
    ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt; TAG=ll1to8; F=$CK/rarecache_ll1to8/fit.pt ;;
    xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; export PREFILL_SMALL_DIR=$M/Qwen3-1.7B
             P=$CK/q17toll8_mix8k/projector.pt; TAG=q17toll8; F=$CK/rarecache_q17toll8/fit.pt; SMALL="" ;;
    xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt; TAG=ll1toq8; F=$CK/rarecache_ll1toq8/fit.pt; SMALL="" ;;
    *) echo "unknown PAIR $PAIR"; return 1 ;;
  esac
  [ -f $P ] || { echo "missing projector $P"; return 1; }
  R=results/fused_$TAG
  CFG=big,${SMALL}hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
  MQ=big,${SMALL}hyb,ours:0.1:sent,ours-acc:0.1:sent,ours-acc:0.03:sent,ours-acc:0.02:sent,ours-acc:0.01:sent,pre:0.1:sent,pre:0.1:sent:0.03,ours-each:0.1:sent,spec-agn:0.1,spec-agn:0.2,spec-agn:0.3,spec-q1:0.1,spec-q1:0.2,spec-each:0.1
  MQSPEED=big,hyb,ours:0.1:sent,ours-acc:0.03:sent,ours-acc:0.01:sent,pre:0.1:sent,pre:0.1:sent:0.03,spec-agn:0.1,spec-q1:0.1,spec-each:0.1
  LB=big,${SMALL}hyb,attn_small:0.1:sent,drop-attn_small:0.1:sent,specprefill:0.1,specprefill:0.3
  QMQ=big,hyb,ours:0.1:sent,ours-acc:0.03:sent,ours-acc:0.01:sent,ours-each:0.1:sent,spec-agn:0.1,spec-agn:0.3,spec-q1:0.1,spec-each:0.1
  echo "PAIR=$PAIR small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} data=${PREFILL_DATA_DIR:-default} projector=$P fold=$PREFILL_FOLD"
}

run_step() {  # $1 pair, $2 step; runs in a subshell so the pair's environment does not leak
  ( setup_pair $1 || exit 1
    STEP=$2
    step "$PAIR / $STEP"
    case $STEP in
      speed)     ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph --configs hyb \
                   --speed_configs $CFG --speed_lengths $LEN --force ;;
      speed_eager) export PREFILL_FOLD=eager  # ours without the fused kernel (large alone again as the in-run reference)
                 ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed_eagerfold --tasks speed_allgraph --configs hyb \
                   --speed_configs big,hyb,attn_small:0.1:sent,attn_small:0.2:sent --speed_lengths $LEN --force ;;
      main)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs $CFG --force ;;
      long)      ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_long --tasks niah_long,qa_long \
                   --configs big,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.3 --force ;;
      multiq)    ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_multiq --tasks qa,speed --configs $MQ --speed_configs $MQSPEED --force ;;
      longbench) ev python scripts/eval_longbench.py --projector $P --out_dir ${R}_longbench --configs $LB --force ;;
      rare)      [ -f $F ] || ev python scripts/fit_rarecache.py --out $F
                 ev python scripts/eval_recompute.py --projector $P --rarecache_fit $F --out_dir ${R}_rare \
                   --tasks speed,niah,ruler,qa,gsm8k,code --configs $RC \
                   --speed_configs big,hyb,attn_small:0.1:sent,specprefill:0.1,specprefill:0.3,$RC --speed_graph --speed_lengths $LEN --force ;;
      quality)   ev python scripts/prepare_quality.py
                 ev python scripts/eval_recompute.py --projector $P --out_dir results/quality_$PAIR --tasks quality --configs $CFG --force
                 ev python scripts/eval_multiq.py --projector $P --out_dir results/quality_mq_$PAIR --pool quality_multiq --tasks qa --configs $QMQ --force ;;
      bench)     PAIR=$PAIR START_AT=short FORCE=1 bash slurm/eval_bench.slurm || { FAILED="$FAILED $PAIR:bench"; echo "!!!!!!!! FAILED: bench $PAIR"; } ;;
      *) echo "unknown step $STEP"; FAILED="$FAILED $PAIR:$STEP" ;;
    esac
    [ -z "$FAILED" ] || echo "FAILED_IN_SUBSHELL:$FAILED" )
}
run() {  # wrapper: collect failures reported by the subshell
  out=$(run_step "$@" 2>&1 | tee /dev/stderr | grep "^FAILED_IN_SUBSHELL:" | tail -1)
  [ -z "$out" ] || FAILED="$FAILED ${out#FAILED_IN_SUBSHELL:}"
}

started=${START_PAIR:+0}; started=${started:-1}
step "phase A: TTFT with every method CUDA-graphed, all pairs"
for p in $PAIRS; do
  [[ $started == 1 || $p == "$START_PAIR" ]] && started=1 || continue
  run $p speed
  run $p speed_eager
done
python scripts/summarize_speed.py  # -> results/speed_summary.md (large / small / SpecPrefill / ours with and without the fused kernel)
step "phase B: everything else, pair by pair"
started=${START_PAIR:+0}; started=${started:-1}
for p in $PAIRS; do
  [[ $started == 1 || $p == "$START_PAIR" ]] && started=1 || continue
  sstarted=${START_STEP:+0}; sstarted=${sstarted:-1}
  for s in main long multiq longbench rare quality bench; do
    if [[ $sstarted == 0 ]]; then [[ $s == "$START_STEP" && $p == "$START_PAIR" ]] && sstarted=1 || continue; fi
    run $p $s
  done
done
python scripts/summarize_speed.py
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
