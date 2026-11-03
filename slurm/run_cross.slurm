#!/bin/bash
#SBATCH --job-name=pf_cross
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
# Cross-family pair (different tokenizers; C2C token alignment, prefill/tokalign.py): projector + the benchmark of
# slurm/run_llama.slurm, one L40S. The large model's data tree is used as is (pools and self-distillation answers are
# in its tokenizer), so no data preparation is needed.
#   sbatch -J pf_xq17ll8 --export=ALL,PAIR=q17_ll8 slurm/run_cross.slurm   Qwen3-1.7B           -> Llama-3.1-8B-Instruct
#   sbatch -J pf_xll1q8  --export=ALL,PAIR=ll1_q8  slurm/run_cross.slurm   Llama-3.2-1B-Instruct -> Qwen3-8B
# Steps as in run_llama.slurm (smoke, stage1, stage2, eval_v2, main, speed, long, multiq, lb, rare); the "small alone"
# config is left out (the small model cannot answer prompts of the other tokenizer; its own numbers are in the
# same-family runs). Restart: sbatch ... --export=ALL,PAIR=...,START_AT=<step>
set -o pipefail
source /nfs/turbo/umms-drjieliu/usr/shengkai/prefillLLM/slurm/env.sh
case ${PAIR:?PAIR=q17_ll8 or ll1_q8} in
  q17_ll8) export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Qwen3-1.7B PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Llama-3.1-8B-Instruct
           export PREFILL_DATA_DIR=$PREFILL_ROOT/data_llama; TAG=q17toll8 ;;
  ll1_q8)  export PREFILL_SMALL_DIR=$PREFILL_ROOT/models/Llama-3.2-1B-Instruct PREFILL_LARGE_DIR=$PREFILL_ROOT/models/Qwen3-8B
           TAG=ll1toq8 ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
echo "PAIR=$PAIR small=$PREFILL_SMALL_DIR large=$PREFILL_LARGE_DIR data=${PREFILL_DATA_DIR:-default}"
START_AT=${START_AT:-smoke}
run=0
step() { echo; echo "==================== [$(date '+%F %T')] $1 ===================="; }
should() { if [[ $run == 1 || $1 == "$START_AT" ]]; then run=1; return 0; fi; return 1; }
CK=$PREFILL_ROOT/checkpoints
SM=$CK/${TAG}_smoke
S1=$CK/${TAG}_stage1
S2=$CK/${TAG}_mix8k
P=$S2/projector.pt
R=results/$TAG
CFG=big,hyb,attn_small:0.1:sent,attn_small:0.2:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.2,specprefill:0.3
RC=rarecache-s:0,rarecache-s:0.1,rarecache-s:0.3,rarecache-s:0.5
FAILED=""
ev() { "$@" || { FAILED="$FAILED $1:$3"; echo "!!!!!!!! FAILED: $*"; }; }  # isolated eval step

if should smoke; then
  set -e
  step "smoke: CPU cross-tokenizer test, then a few GPU training steps and tiny eval runs"
  (cd tests && python smoke_cross.py 2>&1 | grep -E "^\[ok\]|^\[skip\]|PASSED|Error|assert" | tail -12)
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
  step "stage1: representation alignment, in_norm"
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
  ev python scripts/evaluate_v2.py --projectors $TAG=$P --baselines "" --tasks kl,curve --curve_runs $TAG=$S2 --out_dir ${R}_v2
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
    --configs big,hyb,attn_small:0.1:sent,drop-attn_small:0.2:sent,specprefill:0.1,specprefill:0.3
fi
if should multiq; then
  step "eval multiq: shared document, several questions"
  MQ=big,hyb,ours:0.1:sent,ours-acc:0.1:sent,ours-acc:0.03:sent,ours-acc:0.02:sent,ours-acc:0.01:sent,pre:0.1:sent,pre:0.1:sent:0.03,ours-each:0.1:sent,spec-agn:0.1,spec-agn:0.2,spec-agn:0.3,spec-q1:0.1,spec-q1:0.2,spec-each:0.1
  ev python scripts/eval_multiq.py --projector $P --out_dir ${R}_multiq --tasks qa,speed --configs $MQ \
    --speed_configs big,hyb,ours:0.1:sent,ours-acc:0.03:sent,ours-acc:0.01:sent,pre:0.1:sent,pre:0.1:sent:0.03,spec-agn:0.1,spec-q1:0.1,spec-each:0.1
fi
if should lb; then
  step "eval LongBench + 32-shot GSM8K"
  ev python scripts/eval_longbench.py --projector $P --out_dir ${R}_longbench --configs big,hyb,attn_small:0.1:sent,drop-attn_small:0.1:sent,specprefill:0.1,specprefill:0.3
fi
if should rare; then
  step "eval RaReCache (sink/tail forced)"
  F=$CK/rarecache_$TAG/fit.pt
  [ -f $F ] || ev python scripts/fit_rarecache.py --out $F
  ev python scripts/eval_recompute.py --projector $P --rarecache_fit $F --out_dir ${R}_rare \
    --tasks speed,niah,ruler,qa,gsm8k,code --configs $RC \
    --speed_configs big,hyb,attn_small:0.1:sent,specprefill:0.1,specprefill:0.3,$RC --speed_graph --speed_lengths 4096,8192,16384,32768
fi
step "all done${FAILED:+ — FAILED:$FAILED}"
[ -z "$FAILED" ]
