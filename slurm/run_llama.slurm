#!/bin/bash
#SBATCH --job-name=pf_llama
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/logs/%x-%j.out
#SBATCH --mail-user=shengkai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Llama pair, step 2 of 2: Llama-3.2-1B-Instruct -> Llama-3.1-8B-Instruct projector + the current benchmark, one L40S.
# Recipe = the Qwen 1.7B->8B normA recipe (round 5: in_norm stage 1 8000 steps, then round-2 mix8k stage 2 12000 steps),
# which is the round-2 0.6B recipe plus RMS-normalised small-model inputs (needed when the small model's hidden states
# have a different scale; neutral otherwise).
#   smoke   : a few training steps + tiny runs of every eval script, to fail fast
#   stage1  : -> $PREFILL_ROOT/checkpoints/ll1to8_stage1
#   stage2  : -> $PREFILL_ROOT/checkpoints/ll1to8_mix8k
#   eval_v2 : KL per data type, KL + retrieval at every checkpoint            -> results/ll1to8_v2
#   main    : NIAH / RULER / QA / GSM8K / code, 9 configs incl. SpecPrefill  -> results/ll1to8_main
#   speed   : TTFT, every method CUDA-graphed, 4k-32k                         -> results/ll1to8_speed
#   long    : QA + NIAH at 16k / 32k                                          -> results/ll1to8_long
#   multiq  : shared document, several questions                             -> results/ll1to8_multiq
#   lb      : LongBench + 32-shot GSM8K                                       -> results/ll1to8_longbench
#   rare    : RaReCache fit + rarecache-s 0/10/30/50%                         -> results/ll1to8_rare
# Training steps stop the job on failure; eval steps are isolated. Restart: sbatch --export=ALL,START_AT=<step> ...
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/llama_env.sh
START_AT=${START_AT:-smoke}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
CK=$PREFILL_ROOT/checkpoints
SM=$CK/ll1to8_smoke
S1=$CK/ll1to8_stage1
S2=$CK/ll1to8_mix8k
P=$S2/projector.pt
R=results/ll1to8
CFG=big,small,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
RC=rarecache-s:0,rarecache-s:0.1,rarecache-s:0.3,rarecache-s:0.5
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$3"; echo "!!!!!!!! FAILED: $*"; }; }  # isolated eval step

if should smoke; then
  set -e
  step "smoke: CPU unit tests on tiny Llama models, then a few GPU training steps and tiny eval runs"
  (cd tests && SMOKE_ARCH=llama python smoke_rarecache.py 2>&1 | grep -E "^\[ok\]|PASSED|Error|assert" | tail -5)
  rm -rf $SM
  python scripts/train_stage1.py --in_norm --out_dir $SM/s1 --max_steps 6 --accum 2 --val_every 3 --val_seqs 2 --save_every 1000
  python scripts/train_stage2.py --init $SM/s1/projector.pt --out_dir $SM/s2 --max_steps 4 --accum 2 \
    --val_every 2 --val_per_type 1 --save_every 1000 --save_step_every 0 --sd sd_mix4k,sd_mixlong --max_prompt 8064 --curriculum_frac 0
  Q=$SM/s2/projector.pt
  python scripts/eval_recompute.py --projector $Q --out_dir $SM/e_main --tasks niah,gsm8k,code,speed_allgraph,qa_long \
    --configs big,hyb,attn_small:0.1:sent,specprefill:0.1 --speed_configs big,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1 \
    --speed_lengths 32768 --speed_reps 1 --n_gsm8k 2 --n_code 2 --n_qa_long 1
  python scripts/eval_multiq.py --projector $Q --out_dir $SM/e_multiq --tasks speed --speed_lengths 16384 --speed_reps 1
  python scripts/eval_longbench.py --projector $Q --out_dir $SM/e_lb --tasks lcc,gsm32 --n_code 1 --n_gsm 1 --configs hyb,specprefill:0.1
  python scripts/fit_rarecache.py --n_tokens 4000 --out $SM/rare/fit.pt
  python scripts/eval_recompute.py --projector $Q --rarecache_fit $SM/rare/fit.pt --out_dir $SM/e_rare --tasks gsm8k --n_gsm8k 2 \
    --configs rarecache-s:0.1
  grep -h "gsm8k\]" -r $SM/e_main/summary.md | head -3 || true
  set +e
fi
if should stage1; then
  step "stage1: Llama-3.2-1B -> Llama-3.1-8B representation alignment, in_norm"
  python scripts/train_stage1.py --in_norm --out_dir $S1 --max_steps 8000 --accum 4 --lr 2e-4 --time_budget_hours 22 \
    --resume $S1/projector.pt || exit 1
fi
if should stage2; then
  step "stage2: mixed data, prompts <= 8k (round-2 mix8k recipe)"
  python scripts/train_stage2.py --init $S1/projector.pt --max_steps 12000 --accum 4 --lr 5e-5 --warmup 100 \
    --val_every 500 --val_per_type 2 --save_every 500 --save_step_every 2000 --time_budget_hours 40 \
    --sd sd_mix4k,sd_mixlong --max_prompt 8064 --out_dir $S2 || exit 1
fi
if should eval_v2; then
  step "eval: KL per data type; KL + retrieval at every checkpoint"
  ev python scripts/evaluate_v2.py --projectors ll1to8=$P --baselines "" --tasks kl,curve --curve_runs ll1to8=$S2 --out_dir ${R}_v2
fi
if should main; then
  step "eval main: NIAH / RULER / QA / GSM8K / code"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_main --tasks niah,ruler,qa,gsm8k,code --configs $CFG
fi
if should speed; then
  step "eval speed: every method CUDA-graphed"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_speed --tasks speed_allgraph --configs hyb \
    --speed_configs $CFG --speed_lengths 4096,8192,16384,32768
fi
if should long; then
  step "eval long: QA + NIAH at 16k / 32k"
  ev python scripts/eval_recompute.py --projector $P --out_dir ${R}_long --tasks niah_long,qa_long \
    --configs big,small,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.3
fi
if should multiq; then
  step "eval multiq: shared document, several questions"
  MQ=big,small,hyb,ours:0.1:sent,ours-acc:0.1:sent,ours-acc:0.03:sent,pre:0.1:sent,pre:0.1:sent:0.03,ours-each:0.1:sent,spec-agn:0.1,spec-agn:0.2,spec-agn:0.3,spec-q1:0.1,spec-q1:0.2,spec-each:0.1
  ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_multiq --tasks qa,speed --configs $MQ \
    --speed_configs big,hyb,ours:0.1:sent,ours-acc:0.03:sent,pre:0.1:sent,pre:0.1:sent:0.03,spec-agn:0.1,spec-q1:0.1,spec-each:0.1
fi
if should lb; then
  step "eval LongBench + 32-shot GSM8K"
  ev python scripts/eval_longbench.py --projector $P --out_dir ${R}_longbench
fi
if should rare; then
  step "eval RaReCache (sink/tail forced)"
  F=$CK/rarecache_ll1to8/fit.pt
  [ -f $F ] || ev python scripts/fit_rarecache.py --out $F
  ev python scripts/eval_recompute.py --projector $P --rarecache_fit $F --out_dir ${R}_rare \
    --tasks speed,niah,ruler,qa,gsm8k,code --configs $RC \
    --speed_configs big,hyb,attn_small:0.1:sent,specprefill:0.1,specprefill:0.3,$RC --speed_graph --speed_lengths 4096,8192,16384,32768
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
