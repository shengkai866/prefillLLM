#!/bin/bash
#SBATCH --job-name=pf_xconv
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-reverse/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=FAIL
# CPU only (no GPU): converts the 8B's sd_* stores for both cross-family reverse pairs (scripts/convert_sd_cross.py)
# into data_revll8toq17 / data_rev8toll1; run_reverse.slurm's convert step then only links the rest of the data tree.
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-reverse || exit 1
M=$PREFILL_ROOT/models
python scripts/convert_sd_cross.py --src_dir $PREFILL_ROOT/data_llama --dst_dir $PREFILL_ROOT/data_revll8toq17 \
  --src_tok $M/Llama-3.1-8B-Instruct --dst_tok $M/Qwen3-1.7B > logs/convert_revll8toq17.log 2>&1 &
python scripts/convert_sd_cross.py --src_dir $PREFILL_ROOT/data --dst_dir $PREFILL_ROOT/data_rev8toll1 \
  --src_tok $M/Qwen3-8B --dst_tok $M/Llama-3.2-1B-Instruct > logs/convert_rev8toll1.log 2>&1 &
wait -n; a=$?; wait -n; b=$?
tail -n 4 logs/convert_revll8toq17.log logs/convert_rev8toll1.log
exit $((a + b))
