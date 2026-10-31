#!/bin/bash
#SBATCH --job-name=pf_llprep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# CPU-only (no GPU requested): rebuild every token pool with the Llama 3 tokenizer under $PREFILL_ROOT/data_llama, with
# the same scripts, sizes and seeds as the Qwen tree: stage-1 web tokens (FineWeb-Edu + SlimPajama), stage-2 / eval
# pools (prepare_mix), 16k/32k eval pools, multi-question pool. Self-distillation answers need the 8B on a GPU and are
# generated in slurm/run_llama.slurm. Idempotent: finished pieces are skipped on a rerun.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
# HF streaming datasets can abort at interpreter shutdown (PyGILState_Release, exit 134) after all data was written:
# accept a non-zero exit only if the step's final output exists.
run() { out=$1; shift; "$@" || { rc=$?; [ -e "$out" ] && echo "(exit $rc at shutdown, $out present: ok)" || exit $rc; }; }
D=$PREFILL_DATA_DIR
for m in $PREFILL_SMALL_DIR $PREFILL_LARGE_DIR; do [ -f $m/config.json ] || { echo "missing model $m"; exit 1; }; done
step "stage-1 web tokens"
run $D/slimpajama/train_offsets.npy python scripts/download_and_prepare.py --skip_models --n_train 12000 --n_test 300
step "stage-2 and eval pools"
run $D/pools/niah_eval/meta.jsonl python scripts/prepare_mix.py
step "16k / 32k eval pools"
run $D/pools/niah_eval_long/meta.jsonl python scripts/prepare_long_eval.py
step "multi-question pool"
run $D/pools/multiq_eval/meta.jsonl python scripts/prepare_multiq.py
du -sh $PREFILL_DATA_DIR/* $PREFILL_DATA_DIR/pools/*
step "all done"
