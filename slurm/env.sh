# Shared environment for all prefillLLM jobs (sourced by every slurm script).
export PREFILL_ROOT=/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM
export HF_HOME=$PREFILL_ROOT/hf_cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source /home/shengkai/miniconda3/etc/profile.d/conda.sh
conda activate rosetta
mkdir -p $PREFILL_ROOT
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM
echo "== $(date) host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
