#!/bin/bash
#SBATCH --job-name=pf_r2eval
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=END,FAIL
# Early evaluation of the two finished round-2 runs (mix4k, ctrl4k) while the main job still trains mix8k.
# The main job (run_round2.slurm) later evaluates all runs into results/round2/; this one writes results/round2_partial/.
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
CK=$PREFILL_ROOT/checkpoints
python scripts/evaluate_v2.py --out_dir results/round2_partial \
  --projectors "r1=$CK/stage2/projector.pt,ctrl4k=$CK/r2_ctrl4k/projector.pt,mix4k=$CK/r2_mix4k/projector.pt" \
  --curve_runs "mix4k=$CK/r2_mix4k,ctrl4k=$CK/r2_ctrl4k" --tasks kl,niah,curve,ruler,qa,code,gsm8k
python scripts/plot_round2.py results/round2_partial
