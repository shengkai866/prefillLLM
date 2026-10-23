# prefillLLM — small-model prefill → projector → large-model KV cache

Implementation of `projectTarget.md`: a small Qwen3 model (0.6B or 1.7B) prefills the prompt, a token-wise projector
predicts the residual stream of every large-model layer (Qwen3-8B, or 32B), and the large model's own (frozen)
`input_layernorm → k_proj/v_proj → k_norm → RoPE` turns that into a KV cache. The first `n_sink` (8) and last `n_tail`
(64) prompt tokens are computed by the large model itself. Optionally, **selective recompute** lets the large model
recompute a small set of middle tokens (recommended: 10%, chosen by the small model's attention, whole sentences).
Both LLMs stay frozen; only the projector (~86M params for 0.6B→8B) is trained.

Current status, recommended configuration and open problems: `projectTarget.md` (sections 五/六) and the reports below.

## Results at a glance (0.6B → 8B, recompute 10% by 0.6B attention, sentences)
| | 8B alone | this method | no recompute | 0.6B alone |
|---|---|---|---|---|
| RULER 4k / 8k | 0.970 / 0.918 | 0.897 / 0.846 | 0.736 / 0.640 | 0.771 / 0.704 |
| Doc QA F1 3.5k / 7.5k | 0.772 / 0.702 | 0.720 / 0.665 | 0.645 / 0.601 | 0.285 / 0.248 |
| GSM8K | 0.907 | 0.920 | 0.860 | 0.500 |
| TTFT speedup (H100, 4k / 8k / 32k) | 1× | ~1.8 / 1.8 / 1.5× | 2.1 / 2.4 / 2.7× | ~4× |

Remaining gaps: aggregation (RULER cwe), exact copying among near-duplicate uuids (multikey_3), multi-hop QA,
fabrication rate 7% vs 4%. Details: `REPORT_2026-10-21_06B_to_8B.md`.

## Reports (newest first)
| File | Content |
|---|---|
| `REPORT_2026-10-21_06B_to_8B.md` | 0.6B→8B: quality per task vs 8B, remaining gaps, causes, next steps |
| `REPORT_2026-10-15_summary.md` | all experiments up to round 4 (model pairs, recompute, deep handoff, Pareto figure) |
| `REPORT_2026-10-13_round2.md` | round 2: mixed training data, 4k vs 8k, web-only control |
| `REPORT_2026-10-12_progress.md`, `REPORT_2026-10-12_first_experiment.md` | round 1 (superseded) |

## Layout
```
prefill/            library
  config.py         all paths (scratch root, models, data, checkpoints, results); PREFILL_SMALL_DIR / PREFILL_LARGE_DIR
                    select the model pair, PREFILL_LARGE_MAXMEM splits the large model over GPUs (32B on 2 H100)
  projector.py      Projector (optional per-token input RMSNorm `in_norm` + layer soft-mix + shared trunk
                    + per-layer low-rank correction)
  kv.py             hidden_to_kv, hybrid_prefill, student_forward / teacher_forward, greedy_decode
  losses.py         stage-1 losses (cos + normalised MSE), attention-output matching (stage 1 and stage 2)
  data.py           token stores (flat uint32 + offsets) and samplers
  mixdata.py        round-2 prompt builders (web / qa / code / chat / math / synthetic retrieval) and prompt pools
  models.py         frozen bf16 model loading (SDPA attention), new_projector
  trainutil.py      logging, LR schedule, hybrid KL validation
  fast.py           fast inference path (same math): lean Qwen3 forward, algebraically folded projector->KV,
                    sink+tail fused into one large-model call, CUDA-graph wrapper
  recompute.py      selective recompute: scorers (random / recent / attn_small / attn_big / oracle) and units
                    (tokens, fixed blocks, sentences); chosen tokens run in the same call as sink + tail
  deep.py           deep handoff (large model runs its last k layers on all tokens) — negative result, kept for reference
scripts/
  download_and_prepare.py   Qwen3-0.6B/8B + FineWeb-Edu / SlimPajama subsets (tokenised) + GSM8K   (CPU job)
  download_models.py        extra model repos (Qwen3-1.7B, Qwen3-32B), verifies all shards
  prepare_mix.py            round-2 prompt pools: mixed train pools, held-out test pools, QA / code / NIAH eval sets
  gen_selfdistill.py        large-model responses for stage-2 self-distillation
  train_stage1.py           representation alignment (--in_norm for RMS-normalised small-model inputs)
  train_stage2.py           end-to-end KL distillation (--sd stores, length curriculum, --lambda_attn attention loss,
                            --init / --start_step to continue a stopped run)
  evaluate.py               round 1: big vs small vs hybrid (speed, ppl/KL, per-layer oracle, NIAH, GSM8K)
  evaluate_v2.py            KL per data type, QA F1, RULER, NIAH, GSM8K, code next-line, KL + retrieval per checkpoint
  eval_recompute.py         RULER / QA / NIAH / GSM8K / code / TTFT with and without selective recompute
  eval_deep.py              deep-handoff evaluation
  bench_speed.py            TTFT of the fast path vs the large model (eager and CUDA graph), up to 32k
  plot_round2.py, plot_summary.py, plot_06b_report.py   report figures
slurm/                      one script per round / experiment (see the header comment of each); env.sh is shared
tests/                      CPU tests with tiny random Qwen3 models:
  smoke_test.py             oracle projector reproduces true logits
  smoke_fast.py             fast path == reference path
  smoke_recompute.py        recompute of every middle token == large-model prefill
  smoke_deep.py             deep handoff limits (k = 0 / k = all layers)
  smoke_r5.py               in_norm (reference vs folded path, old checkpoints) and the stage-2 attention loss
```
Models, data, self-distillation stores and checkpoints live under `/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM`
(`models/`, `data/`, `checkpoints/<run>/projector.pt`). Evaluation outputs (`summary.md`, `results.json`,
`generations.jsonl`, `figures/`) are written to `results/<run>/` in this directory; SLURM logs go to `logs/`.

Main checkpoints:
| Run | Pair | Checkpoint | Results |
|---|---|---|---|
| round 2 mix8k (recommended) | 0.6B→8B | `checkpoints/r2_mix8k` | `results/round2`, `results/recompute_blocks`, `results/r5_eval06` |
| round 5 normA / normB | 1.7B→8B, in_norm (B: + attention loss) | `checkpoints/r5_s17_normA`, `r5_s17_normB` | `results/r5_s17_normA{,_v2}`, `results/r5_s17_normB{,_v2}` |
| round 3b / 4 | 1.7B→32B, no in_norm (did not learn retrieval) | `checkpoints/s17to32_mix8k`, `r4_more_17_32` | `results/r3_s17_to_32b{,_v2}`, `results/r4_more_17_32{,_v2}` |
| round 6 (running) | 1.7B→32B, in_norm | `checkpoints/s17to32n_mix8k` | `results/r6_s17_to_32b_norm{,_v2}` |

## Run
Every slurm script runs its steps sequentially in one job and can restart from a step:
```bash
sbatch slurm/run_all.slurm                                   # round 1: data -> stage1 -> stage2 -> eval
sbatch slurm/run_round2.slurm                                # round 2: pools -> self-distillation -> 3 stage-2 runs -> eval -> plots
sbatch slurm/run_r3_small17.slurm                            # round 3a: 1.7B -> 8B
sbatch slurm/run_r6_32b_norm.slurm                           # round 6: 1.7B -> 32B with in_norm (2 H100)
sbatch --export=ALL,START_AT=stage2 slurm/run_r6_32b_norm.slurm   # restart from a step (names in the script header)
sbatch slurm/eval_recompute_blocks.slurm                     # selective-recompute quality sweep (no training)
```
8B pairs fit on one GPU (`drjieliu-l40s` or `drjieliu-h100`; the stage-2 attention loss needs an H100). 32B training
needs 2 H100s; 32B evaluation fits on one. Run the CPU tests with `python tests/<name>.py`.

## Using a trained projector
```python
from prefill.models import load_pair
from prefill.projector import Projector
from prefill.kv import hybrid_prefill, greedy_decode
tok, small, large = load_pair("cuda")
proj, _ = Projector.load(".../checkpoints/r2_mix8k/projector.pt", "cuda"); proj.eval()
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
cache, logits = hybrid_prefill(large, small, proj, ids, n_sink=8, n_tail=64)   # TTFT ends here
out = greedy_decode(large, cache, logits, 256, eos_ids=(tok.eos_token_id,))
```

## Fast inference
```python
from prefill.fast import FastQwen3, FoldedProjectorKV, FastHybrid, Graphed, kv_to_dynamic_cache
S, L = FastQwen3(small), FastQwen3(large)          # re-points HF weights to fused views, no extra memory
fh = FastHybrid(S, L, FoldedProjectorKV(proj, L), n_sink=8, n_tail=64)
kbuf, vbuf, logits = fh.prefill(ids_1d)            # or Graphed(fh.prefill, ids_1d) for a fixed length
out = greedy_decode(large, kv_to_dynamic_cache(kbuf, vbuf), logits, 256, eos_ids=(tok.eos_token_id,))
```
Selective recompute is built on the same fast path (`prefill/recompute.py`, `RecomputeHybrid`); configuration strings
used by `scripts/eval_recompute.py` look like `hyb`, `attn_small:0.1:sent`, `attn_big:0.05`, `random:0.1`,
`attn_big:0.1:b32` (scorer : fraction of middle tokens : unit).

## Training data (round 2 onwards)
Web 25%, extractive QA 20% (SQuAD v2, HotpotQA, TriviaQA), code 15% (codeparrot), chat 15% (UltraChat),
math 10% (MetaMathQA few-shot), synthetic retrieval 15% (records, JSON uuid key-value, verbatim sentence copy; templates
differ from the NIAH / RULER tests). Responses are generated by the large model (self-distillation). Tests use held-out
material only (validation sets, held-out repos, hash-split MetaMath) plus RULER 4k/8k (`simonjegou/ruler`) and GSM8K.

## Notes / deviations from the plan
* `cerebras/SlimPajama-627B` is no longer reachable on the Hub (401); `DKYoon/SlimPajama-6B` (public subsample) is used.
* Hidden states are computed online (not cached to disk): one 4k sequence of 8B hidden states is 1.2 GB.
* 1.7B→32B reuses the 8B-generated self-distillation responses (teacher logits come from the 32B).
* `train_stage2.py` does not save optimizer state; `--start_step` continues the LR schedule and curriculum, with a short
  re-warmup. Do not re-warm an already trained projector to a high LR: it destroys learned retrieval.
* Not implemented: anchor layers, LoRA on the large model, vLLM/SGLang integration, CUDA graphs for the recompute path.
