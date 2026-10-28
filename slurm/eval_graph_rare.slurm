#!/bin/bash
#SBATCH --job-name=pf_grare
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Two follow-ups of REPORT_2026-10-28_followup.md on ONE L40S, both small models (0.6B / r2_mix8k, 1.7B / r5_s17_normB) -> 8B.
#   A. fair speed: TTFT with EVERY method as piecewise CUDA graphs (task speed_allgraph, prefill/graphs.py): 8B alone,
#      small alone, hyb, ours 10% / 20%, drop-rest 20%, SpecPrefill 10 / 20 / 30%; 4k-32k
#      -> results/speed_allgraph_06|17 (~20 min each). Until now only SpecPrefill's look-ahead was graphed.
#   B. RaReCache rerun with the sink 8 / tail 64 forced (rarecache-s:*; results/rare_diag_06 showed the paper protocol's
#      unforced attention sink is what breaks it): rho 0/10/20/30/50% on speed, NIAH, RULER 4k/8k, QA, GSM8K, code,
#      same examples as results/rare_06|17 and spec_06|17 -> results/rare_s_06|17 (~4 h each; existing fit.pt reused).
# Order: A 0.6B, A 1.7B, B 0.6B, B 1.7B. Each part is isolated (a failure does not stop the others); restarting skips
# finished tasks.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
FAILED=""

GCFG=big,small,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
R=rarecache-s:0,rarecache-s:0.1,rarecache-s:0.2,rarecache-s:0.3,rarecache-s:0.5
SPEED="--speed_configs big,hyb,attn_small:0.1:sent,attn_small:0.2:sent,specprefill:0.1,specprefill:0.3,$R --speed_graph --speed_lengths 4096,8192,16384,32768"

step "smoke tests (CPU, tiny models)"
(cd tests && python smoke_rarecache.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert" | tail -4)

part_a() {  # $1 tag, $2 small model dir, $3 projector
  set -e
  export PREFILL_SMALL_DIR=$M/$2
  step "A $2: speed, every method CUDA-graphed"
  python scripts/eval_recompute.py --projector $CK/$3/projector.pt --out_dir results/speed_allgraph_$1 \
    --tasks speed_allgraph --configs hyb --speed_configs $GCFG --speed_lengths 4096,8192,16384,32768
}

part_b() {  # $1 tag, $2 small model dir, $3 projector
  set -e
  export PREFILL_SMALL_DIR=$M/$2
  step "B $2: RaReCache with sink/tail forced"
  python scripts/fit_rarecache.py  # skips when fit.pt exists
  python scripts/eval_recompute.py --projector $CK/$3/projector.pt --rarecache_fit $CK/rarecache_${2#Qwen3-}/fit.pt \
    --out_dir results/rare_s_$1 --tasks speed,niah,ruler,qa,gsm8k,code --configs $R $SPEED
}

# Run each part as a plain statement: inside "( f ) || ..." bash would silently ignore the set -e in f.
( part_a 06 Qwen3-0.6B r2_mix8k ); [ $? -eq 0 ] || { FAILED="$FAILED A-0.6B"; step "A 0.6B FAILED, continuing"; }
( part_a 17 Qwen3-1.7B r5_s17_normB ); [ $? -eq 0 ] || { FAILED="$FAILED A-1.7B"; step "A 1.7B FAILED, continuing"; }
( part_b 06 Qwen3-0.6B r2_mix8k ); [ $? -eq 0 ] || { FAILED="$FAILED B-0.6B"; step "B 0.6B FAILED, continuing"; }
( part_b 17 Qwen3-1.7B r5_s17_normB ); [ $? -eq 0 ] || FAILED="$FAILED B-1.7B"
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
