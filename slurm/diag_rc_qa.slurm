#!/bin/bash
#SBATCH --job-name=pf_rcqa
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Branch moe-router: QA benchmark diagnostic of the recompute-aware projectors (scripts/diag_rc_qa.py) for all five pairs,
# one after the other on one L40S -> results/rc_qa_<tag>/per_example.jsonl
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
M=$PREFILL_ROOT/models
for PAIR in ${PAIRS:-06 17 ll xll1q8 xq17ll8}; do
  ( unset PREFILL_SMALL_DIR PREFILL_LARGE_DIR PREFILL_DATA_DIR
    case $PAIR in
      06) export PREFILL_SMALL_DIR=$M/Qwen3-0.6B; P=r2_mix8k; TAG=06 ;;
      17) export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=r5_s17_normB; TAG=17 ;;
      ll) source slurm/llama_env.sh; P=ll1to8_mix8k; TAG=ll1to8 ;;
      xq17ll8) source slurm/llama_env.sh; export PREFILL_SMALL_DIR=$M/Qwen3-1.7B; P=q17toll8_mix8k; TAG=q17toll8 ;;
      xll1q8) export PREFILL_SMALL_DIR=$M/Llama-3.2-1B-Instruct; P=ll1toq8_mix8k; TAG=ll1toq8 ;;
    esac
    echo "==================== [$(date '+%F %T')] $PAIR"
    python scripts/diag_rc_qa.py --base $CK/$P/projector.pt --rc $CK/rc_$TAG/projector.pt --out_dir results/rc_qa_$TAG
  ) || echo "!!!!!!!! FAILED: $PAIR"
done
