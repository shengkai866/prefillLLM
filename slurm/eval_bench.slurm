#!/bin/bash
#SBATCH --job-name=pf_bench
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Long-context benchmark suite (scripts/prepare_bench.py + scripts/eval_bench.py) for one trained pair:
#   LastingBench, MuSiQue, LooGLE, LooGLE-long, Qasper, CLIPPER, LongHealth (fold 0), ContractNLI; QuALITY comes
#   from slurm/eval_quality.slurm. Configs: large model alone, ours (attn_small:0.1:sent), SpecPrefill 10 % / 20 %.
#   -> results/bench_<PAIR>/{results.json,summary.md,generations.jsonl}; results/bench_summary.md over all pairs.
#   sbatch --export=ALL,PAIR=06 slurm/eval_bench.slurm        Qwen3-0.6B -> Qwen3-8B   (r2_mix8k)        L40S
#   sbatch --export=ALL,PAIR=17 ...                           Qwen3-1.7B -> Qwen3-8B   (r5_s17_normB)    L40S
#   sbatch --export=ALL,PAIR=ll ...                           Llama-3.2-1B -> Llama-3.1-8B (ll1to8_mix8k) L40S
#   sbatch --partition=drjieliu-h100 --export=ALL,PAIR=32 ... Qwen3-1.7B -> Qwen3-32B  (s17to32n_mix8k)  H100, 16k budget
#   sbatch --export=ALL,PAIR=xq17ll8 | xll1q8 ...             cross-family pairs
# Steps (START_AT=<step> restarts from one): pools, smoke, short (prompts <= 32k: no YaRN), long (LooGLE / CLIPPER:
# YaRN for Qwen3), summary. MAX_LEN overrides the prompt budget (default 65536; 32B: 16384).
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
MAX_LEN=${MAX_LEN:-65536}
case ${PAIR:?PAIR=06|17|ll|32|xq17ll8|xll1q8} in
  06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=$CK/r2_mix8k/projector.pt ;;
  17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/r5_s17_normB/projector.pt ;;
  32) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-32B; P=$CK/s17to32n_mix8k/projector.pt; MAX_LEN=${MAX_LEN_32:-16384} ;;
  ll) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh; P=$CK/ll1to8_mix8k/projector.pt ;;
  xq17ll8) source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
           export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=$CK/q17toll8_mix8k/projector.pt ;;
  xll1q8)  export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=$CK/ll1toq8_mix8k/projector.pt ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
[ -f $P ] || { echo "missing projector $P"; exit 1; }
echo "PAIR=$PAIR small=${PREFILL_SMALL_DIR:-default} large=${PREFILL_LARGE_DIR:-default} projector=$P max_len=$MAX_LEN"
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
OUT=results/bench_$PAIR
CFG=${CFG:-big,attn_small:0.1:sent,specprefill:0.1,specprefill:0.2}
SHORT=lastingbench,musique,qasper,longhealth,contractnli
LONG=loogle,loogle_long,clipper
START_AT=${START_AT:-pools}
after() { [[ $(printf "pools\nsmoke\nshort\nlong\nsummary\n" | grep -n "^$START_AT$" | cut -d: -f1) -le $(printf "pools\nsmoke\nshort\nlong\nsummary\n" | grep -n "^$1$" | cut -d: -f1) ]]; }
FAILED=""
ev() { "$@" || { FAILED="$FAILED $2"; echo "!!!!!!!! FAILED: $*"; }; }

if after pools; then
  step "pools (CPU, large model's tokenizer)"
  python scripts/prepare_bench.py --max_len $MAX_LEN || exit 1
fi
if after smoke; then
  step "smoke: 2 items per task"
  python scripts/prepare_bench.py --max_len $MAX_LEN --n 2 --suffix _smoke && \
  python scripts/eval_bench.py --projector $P --out_dir results/bench_smoke_$PAIR --max_len $MAX_LEN --suffix _smoke --configs big,attn_small:0.1:sent,specprefill:0.1 --force || exit 1
fi
if after short; then
  step "short tasks ($SHORT)"
  ev python scripts/eval_bench.py --projector $P --out_dir $OUT --max_len $MAX_LEN --tasks $SHORT --configs $CFG
fi
if after long; then
  step "long tasks ($LONG)"
  ev python scripts/eval_bench.py --projector $P --out_dir $OUT --max_len $MAX_LEN --tasks $LONG --configs $CFG
fi
step "summary over all pairs"
python scripts/summarize_bench.py
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
