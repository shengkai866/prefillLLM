#!/bin/bash
#SBATCH --job-name=pf_c2cdry
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-c2c/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=FAIL
# CPU-only dry run of C2C's official trainer (scripts/c2c_official_train.py): builds the OpenHermes-2.5 datasets cache
# (too big for the login node) and checks that our loader (prefill/c2c.py from_rosetta_dir) reads the checkpoint layout
# the trainer really writes. Tiny pair Qwen3-1.7B -> Qwen3-0.6B, 40 samples, length 2048 (a length of 256 truncates whole prompts away and the batch then has no fused section, so no grad).
set -eo pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
cd /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM-c2c
export PREFILL_C2C_DIR=$PREFILL_ROOT/ext/C2C CUDA_VISIBLE_DEVICES=""
M=$PREFILL_ROOT/models
D=$PREFILL_ROOT/checkpoints/c2c_dryrun
rm -rf $D; mkdir -p $D
cat > $D/recipe.json <<EOF
{"model": {"base_model": "$M/Qwen3-0.6B", "teacher_model": "$M/Qwen3-1.7B", "is_do_alignment": false,
  "alignment_strategy": "first", "projector": {"type": "C2CProjector", "params": {"hidden_dim": 1024,
  "intermediate_dim": 1024, "num_layers": 3, "dropout": 0.1, "initial_temperature": 1.0, "final_temperature": 0.001,
  "anneal_steps": 4}}, "mapping": "last_aligned"},
 "training": {"learning_rate": 1e-4, "weight_decay": 0.01, "num_epochs": 1, "max_length": 2048, "device": "cpu",
  "scheduler_type": "linear", "warmup_ratio": 0.1, "max_grad_norm": 1.0, "gradient_accumulation_steps": 4,
  "per_device_train_batch_size": 2, "num_processes": 1, "freeze": ["teacher", "base"], "seed": 42},
 "output": {"output_dir": "$D", "save_steps": 2, "eval_steps": 2,
  "wandb_config": {"project": "Rosetta", "mode": "disabled", "entity": null, "run_name": "dry"}},
 "data": {"type": "OpenHermesChatDataset", "kwargs": {"split": "train", "max_word_count": 2048, "num_samples": 40},
  "train_ratio": 0.9}}
EOF
python scripts/c2c_official_train.py --config $D/recipe.json
ls -la $D $D/final
PREFILL_SMALL_DIR=$M/Qwen3-1.7B PREFILL_LARGE_DIR=$M/Qwen3-0.6B python - <<EOF
import torch
from prefill.projector import Projector
f, ck = Projector.load("$D/final")
print("loaded", type(f).__name__, "offset", f.offset, "layers", len(f.projs), "params", f.num_params(), ck)
print("gates", f.gate_logits()[:3])
EOF
echo DRYRUN OK
