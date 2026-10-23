#!/bin/bash
# Submit the whole pipeline as a dependency chain: download -> stage1 -> stage2 -> eval.
# Usage: bash slurm/submit_all.sh            (full chain)
#        bash slurm/submit_all.sh --from 2   (start at stage1, e.g. data already prepared)
set -e
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM
FROM=1
[[ "$1" == "--from" ]] && FROM=$2
dep=""
sub() { # sub <script>
  if [[ -n "$dep" ]]; then jid=$(sbatch --parsable --dependency=afterok:$dep "$1"); else jid=$(sbatch --parsable "$1"); fi
  echo "submitted $1 -> job $jid (after: ${dep:-none})"; dep=$jid
}
[[ $FROM -le 1 ]] && sub slurm/01_download.slurm
[[ $FROM -le 2 ]] && sub slurm/02_stage1.slurm
[[ $FROM -le 3 ]] && sub slurm/03_stage2.slurm
[[ $FROM -le 4 ]] && sub slurm/04_eval.slurm
squeue -u $USER -o "%.10i %.12j %.28P %.8T %.10M %.6D %R"
