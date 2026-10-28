#!/bin/bash
#SBATCH --job-name=pf_follow
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=40:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Follow-up experiments after REPORT_2026-10-27_specprefill.md, ONE L40S GPU, no training. Four independent parts;
# a failed part does not stop the next, the job exits non-zero if any failed. Restarting skips finished tasks.
#   A  fair speed: SpecPrefill look-ahead as a CUDA graph (--speed_graph) + where the hybrid's time goes (breakdown),
#      both small models -> results/speed_graph_06|17; and 16k / 32k quality (qa_long, niah_long) for 0.6B -> results/long_06
#   B  shared document, several questions (prefill/multiq.py, scripts/eval_multiq.py), both -> results/multiq_06|17
#   C  RaReCache baseline (prefill/rarecache.py): fit + eval, both -> results/rare_06|17 (same as eval_rarecache.slurm)
#   D  whole-context tasks (scripts/eval_longbench.py): LongBench summarisation / code + 32-shot GSM8K, 0.6B -> results/longbench_06
# Expected: A ~1.5 h, B ~3.5 h, C ~6.5 h, D ~4 h.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
P06="06 Qwen3-0.6B r2_mix8k"
P17="17 Qwen3-1.7B r5_s17_normB"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

step "smoke tests (CPU, tiny models)"
(cd tests && python smoke_multiq.py 2>&1 | grep -E "PASSED|FAILED|Error|assert" | tail -3)
(cd tests && python smoke_rarecache.py 2>&1 | grep -E "PASSED|FAILED|Error|assert" | tail -3)

part_speed() {
  set -e
  Q=big,small,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3,specprefill:0.1:lah1
  for P in "$P06" "$P17"; do
    set -- $P
    step "A $2: speed with graphed look-ahead + breakdown"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/speed_graph_$1 \
      --tasks speed,breakdown --speed_configs $Q --speed_graph --speed_lengths 4096,8192,16384,32768
  done
  set -- $P06
  step "A $2: QA + NIAH at 16k / 32k"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/long_$1 \
    --tasks niah_long,qa_long --configs big,small,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.3
}

part_multiq() {
  set -e
  for P in "$P06" "$P17"; do
    set -- $P
    step "B $2: shared document, several questions"
    PREFILL_SMALL_DIR=$M/$2 python scripts/eval_multiq.py --projector $CK/$3/projector.pt --out_dir results/multiq_$1 --tasks qa,speed
  done
}

part_rare() {
  set -e
  R=rarecache:0,rarecache:0.1,rarecache:0.2,rarecache:0.3,rarecache:0.5
  SPEED="--speed_configs big,hyb,attn_small:0.1:sent,attn_small:0.2:sent,specprefill:0.1,specprefill:0.3,$R --speed_lengths 4096,8192,16384,32768"
  for P in "$P06" "$P17"; do
    set -- $P
    export PREFILL_SMALL_DIR=$M/$2
    step "C $2: fit RaReCache maps"
    python scripts/fit_rarecache.py
    step "C $2: speed, NIAH, RULER, QA, GSM8K, code"
    python scripts/eval_recompute.py --projector $CK/$3/projector.pt --rarecache_fit $CK/rarecache_${2#Qwen3-}/fit.pt \
      --out_dir results/rare_$1 --tasks speed,niah,ruler,qa,gsm8k,code --configs $R $SPEED
  done
  unset PREFILL_SMALL_DIR
}

part_longbench() {
  set -e
  set -- $P06
  step "D $2: LongBench summarisation / code + 32-shot GSM8K"
  PREFILL_SMALL_DIR=$M/$2 python scripts/eval_longbench.py --projector $CK/$3/projector.pt --out_dir results/longbench_$1
}

# Run each part as a plain statement: inside "( f ) || ..." bash would silently ignore the set -e in f.
( part_speed );     [ $? -eq 0 ] || { FAILED="$FAILED speed";     step "part A FAILED, continuing"; }
( part_multiq );    [ $? -eq 0 ] || { FAILED="$FAILED multiq";    step "part B FAILED, continuing"; }
( part_rare );      [ $? -eq 0 ] || { FAILED="$FAILED rarecache"; step "part C FAILED, continuing"; }
( part_longbench ); [ $? -eq 0 ] || FAILED="$FAILED longbench"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
