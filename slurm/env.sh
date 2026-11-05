# Shared environment for all prefillLLM jobs (sourced by every slurm script).
export PREFILL_ROOT=/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM
export HF_HOME=$PREFILL_ROOT/hf_cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PREFILL_FOLD=${PREFILL_FOLD:-fused}  # projector -> KV stage: fused Triton kernel (prefill/fold_fused.py) | eager
export TRITON_CACHE_DIR=$PREFILL_ROOT/triton_cache
source /home/shengkai/miniconda3/etc/profile.d/conda.sh
conda activate rosetta
mkdir -p $PREFILL_ROOT
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-moe  # branch moe-router: its own worktree (main = prefillLLM); revert on merge
echo "== $(date) host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION gpus=${CUDA_VISIBLE_DEVICES:-none} fold=$PREFILL_FOLD"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
