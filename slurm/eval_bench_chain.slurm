#!/bin/bash
#SBATCH --job-name=pf_bench_chain
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# One GPU, several pairs in a row: runs slurm/eval_bench.slurm (long-context suite) for every pair in PAIRS, one after
# the other, each restarting from START_AT (default short: finished tasks are skipped by scripts/eval_bench.py, so a
# pair that is already half done only runs what is missing). A failed pair does not stop the chain.
#   sbatch slurm/eval_bench_chain.slurm                                   # PAIRS="ll xq17ll8 06 17 xll1q8"
#   sbatch --export=ALL,PAIRS="06 17" slurm/eval_bench_chain.slurm
# The 32B pair does not fit a 48 GB L40S; queue it on the H100 after this chain:
#   sbatch --partition=drjieliu-h100 --dependency=afterany:<this job id> --export=ALL,PAIR=32,START_AT=short slurm/eval_bench.slurm
set -o pipefail
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM
PAIRS=${PAIRS:-"ll xq17ll8 06 17 xll1q8"}
START_AT=${START_AT:-short}
FAILED=""
for p in $PAIRS; do
  echo; echo "#################### [$(date '+%F %T')] pair $p ####################"
  PAIR=$p START_AT=$START_AT bash slurm/eval_bench.slurm || FAILED="$FAILED $p"
done
echo; echo "#################### [$(date '+%F %T')] chain done${FAILED:+ — FAILED:$FAILED} ####################"
[ -z "$FAILED" ]
