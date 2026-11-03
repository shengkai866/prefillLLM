# prefillLLM — small-model prefill → projector → large-model KV cache

Implementation of `projectTarget.md`. A small model prefills the prompt. A token-wise projector predicts the residual
stream of every large-model layer from the small model's hidden states, and the large model's own frozen
`input_layernorm → k_proj/v_proj → (k_norm) → RoPE` turns that into a KV cache. The large model computes the first
`n_sink` (8) and last `n_tail` (64) prompt tokens itself. **Selective recompute** also lets it recompute a small set of
middle tokens: by default 10%, picked as whole sentences by the small model's attention. Unselected tokens keep their
projected KV instead of being dropped. Both LLMs stay frozen; only the projector is trained (~86M params for 0.6B→8B).

The result is a **complete, question-independent cache**, which matters for tasks that need the whole context and for
asking several questions about one document. The method has been validated on three model pairs (Qwen 0.6B→8B,
Qwen 1.7B→32B, Llama 1B→8B). Two cross-family pairs were still training as of 2026-11-02.

## Status (2026-11-02)
| Pair | Projector | Status | Main report |
|---|---|---|---|
| Qwen3-0.6B → Qwen3-8B | `r2_mix8k` | done | `REPORT_2026-10-21`, `-10-27_*`, `-10-28_followup` |
| Qwen3-1.7B → Qwen3-8B | `r5_s17_normB` (`r5_s17_normA` = same recipe without attention loss) | done | `REPORT_2026-10-27_06B_vs_17B` |
| Qwen3-1.7B → Qwen3-32B | `s17to32n_mix8k` (round 6, `in_norm`) | done | `REPORT_2026-11-02_32B_llama` §2 |
| Llama-3.2-1B → Llama-3.1-8B (Instruct) | `ll1to8_mix8k` | done | `REPORT_2026-11-02_32B_llama` §3 |
| Qwen3-1.7B → Llama-3.1-8B (cross-family) | `q17toll8_mix8k` | running, job 28047524 | `REPORT_2026-11-02_32B_llama` §6 |
| Llama-3.2-1B → Qwen3-8B (cross-family) | `ll1toq8_mix8k` | running, job 28047556 | `REPORT_2026-11-02_32B_llama` §6 |

Checkpoints are under `$PREFILL_ROOT/checkpoints/<name>/projector.pt`.

## Results at a glance
TTFT speedup is relative to the large model alone, with every method CUDA-graphed. "Ours" means recompute 10% of the
middle tokens. The comparison rows say whether ours wins against SpecPrefill at the same speed.

| | Qwen 0.6B → 8B (L40S) | Qwen 1.7B → 32B (H100) | Llama 1B → 8B (L40S) |
|---|---|---|---|
| KL to large model 4k / 8k | 0.136 / — | 0.098 / 0.120 | 0.105 / 0.120 |
| TTFT speedup, no recompute (4k / 8k / 16k / 32k) | 2.78 / 3.16 / 2.97 / 2.92 | 4.58 / 5.39 / 6.16* / — | 2.67 / 3.00 / 3.19 / 3.39 |
| TTFT speedup, ours | 2.29 / 2.31 / 2.06 / 1.87 | 3.04 / 2.87 / 2.73* / — | 2.19 / 2.25 / 2.14 / 2.05 |
| TTFT speedup, SpecPrefill 10% | 2.51 / 2.98 / 2.87 / 2.84 | 3.14 / 3.74 / 4.53* / — | 2.42 / 2.88 / 3.15 / 3.36 |
| RULER 4k, large / ours | 0.970 / 0.897 | 0.976 / 0.914 (20%: 0.973) | 0.974 / 0.928 |
| vs SpecPrefill: RULER 4k / 8k | tie / slightly lower (−0.02) | tie / lower (−0.06) | tie / lower (−0.07) |
| vs SpecPrefill: QA at 4k / 8k | tie | lower | lower |
| vs SpecPrefill: GSM8K few-shot (and `####` format) | win | win | win |
| vs SpecPrefill: LongBench code (LCC / RepoBench-P) | win, +5–12 | win, +4–6 | win, +5–11 |
| vs SpecPrefill: long QA | win at 16k / 32k, +5–6 | win at 16k, +5 | tie at 16k, lose at 32k (−9) |
| Same document, 5 questions: total time vs large model reusing its own cache | 1.2–2.0× faster, F1 ≥ 8B | 1.7–2.7× faster, F1 ≈ 32B | 1.35–1.67× faster, F1 3–4 pts below 8B |
| RaReCache baseline | far below ours | not run (memory) | far below ours |

\* 32B at 16k is eager: the CUDA graphs do not fit on one H100. 32B does not fit 32k on one H100.

The overall picture:
- **Where we win.** Ours wins on tasks that need the full context: few-shot GSM8K, LongBench code completion, and
  several questions about one document. SpecPrefill's cache is selected for one question, so it fails on later
  questions (F1 0.17–0.39) unless it re-prefills for each one.
- **Where we lose.** SpecPrefill is faster for a single question, and the gap grows with length. It is also better on
  RULER 8k aggregation subtasks (cwe / fwe), which are limited by the small model. On Llama it is better on 32k long QA,
  likely because the projector was trained only up to 8k.
- **What made 1.7B work as the small model.** RMS-normalising the small model's inputs (`in_norm`) was required, for 8B
  and 32B targets alike. Without it the projector never learned retrieval.

## Reports (newest first)
| File | Content |
|---|---|
| `REPORT_2026-11-02_32B_llama.md` | 1.7B → 32B; Llama 1B → 8B; cross-family setup (C2C token alignment); appendix mapping **every** `results/` dir to a report |
| `REPORT_2026-10-28_followup.md` | vs SpecPrefill: fair CUDA-graph timing, 16k/32k QA, multi-question reuse (incl. 20 questions), LongBench, RaReCache reproduction and diagnosis |
| `REPORT_2026-10-27_specprefill.md` | first comparison with SpecPrefill (0.6B / 1.7B → 8B), "drop the rest" ablation |
| `REPORT_2026-10-27_06B_vs_17B.md` | 0.6B vs 1.7B as the small model for 8B |
| `REPORT_2026-10-21_06B_to_8B.md` | 0.6B → 8B in depth: per-task gaps vs 8B, causes |
| `REPORT_2026-10-15_summary.md` | rounds 1–4 (model pairs, recompute, deep handoff, Pareto figure) |
| `REPORT_2026-10-13_round2.md` | round 2: mixed training data, 4k vs 8k, web-only control |
| `REPORT_2026-10-12_progress.md`, `REPORT_2026-10-12_first_experiment.md` | round 1 (superseded) |

## Layout
```
prefill/            library
  config.py         paths and model selection, all from env vars (see "Environment" below)
  projector.py      Projector: optional per-token input RMSNorm (`in_norm`) + layer soft-mix + shared trunk
                    + per-layer low-rank correction
  kv.py             hidden_to_kv, hybrid_prefill, student_forward / teacher_forward, greedy_decode (Qwen3 and Llama)
  losses.py         stage-1 losses (cos + normalised MSE), attention-output matching (stage 1 and stage 2)
  data.py           token stores (flat uint32 + offsets) and samplers
  mixdata.py        round-2 prompt builders (web / qa / code / chat / math / synthetic retrieval) and prompt pools
  models.py         frozen bf16 model loading (SDPA; load_split puts a large model on several GPUs), new_projector
  trainutil.py      logging, LR schedule, hybrid KL validation
  fast.py           fast inference path (same math): lean Qwen3/Llama forward, algebraically folded projector->KV,
                    sink+tail fused into one large-model call, CUDA-graph wrapper (Graphed), FastBig
  recompute.py      selective recompute (RecomputeHybrid): scorers random / recent / attn_small / attn_big / oracle /
                    spec; units tokens / fixed blocks / sentences; drop-* ablations; the SpecPrefill baseline
                    (prefill_specprefill, CUDA-graphed look-ahead LookaheadGraph)
  graphs.py         piecewise CUDA graphs of every prefill method, for a fair TTFT comparison
  multiq.py         shared document + several questions (MultiQ): big / hyb / ours / ours-acc (cumulative) / pre /
                    SpecPrefill variants (question-agnostic, first-question, re-prefill per question)
  rarecache.py      RaReCache baseline (ridge maps + rank-disagreement recompute), reimplemented from the paper
  tokalign.py       cross-tokenizer pairs: C2C token alignment as a lookup table over the large vocabulary
  deep.py           deep handoff (large model runs its last k layers on all tokens): negative result, kept for reference
scripts/
  download_and_prepare.py   Qwen3-0.6B/8B + FineWeb-Edu / SlimPajama subsets (tokenised) + GSM8K   (CPU)
  download_models.py        extra model repos (Qwen3-1.7B, Qwen3-32B), verifies all shards
  prepare_mix.py            mixed train pools, held-out test pools, QA / code / NIAH eval sets
  prepare_long_eval.py      16k / 32k QA and NIAH pools
  prepare_multiq.py         shared-document multi-question pools (multiq_eval: 5 questions, multiq_eval20: 20)
  gen_selfdistill.py        large-model responses for stage-2 self-distillation
  train_stage1.py           representation alignment (--in_norm)
  train_stage2.py           end-to-end KL distillation (--sd stores, length curriculum, --lambda_attn,
                            --init / --start_step to continue a stopped run)
  evaluate_v2.py            KL per data type, QA F1, RULER, NIAH, GSM8K, code; KL + retrieval accuracy per checkpoint
  eval_recompute.py         main benchmark: RULER / QA / NIAH / GSM8K / code / 16k-32k / TTFT (eager, graphed,
                            all-graph, stage breakdown) for any config string (below)
  eval_multiq.py            multi-question quality and per-question latency
  eval_longbench.py         LongBench gov_report / QMSum / LCC / RepoBench-P + 32-shot GSM8K
                            (--code_chat --code_prefix '```\n' for Qwen3-32B, see the 11-02 report §2.3)
  fit_rarecache.py, diag_rarecache.py   RaReCache fit and fit diagnostics
  analyze_qa_errors.py      QA error types (refusal / wrong span / fabricated) from generations.jsonl
  tokmap_stats.py           cross-tokenizer alignment loss on real prompts -> results/tokalign
  bench_speed.py            TTFT of the fast path vs the large model, up to 32k
  evaluate.py, eval_deep.py round 1 and deep-handoff evaluation (superseded)
  plot_*.py                 report figures
slurm/                      one script per experiment; the header comment of each lists its steps and outputs.
                            env.sh is shared; llama_env.sh switches to the Llama pair
tests/                      CPU tests with tiny random models (SMOKE_ARCH=llama runs smoke_fast / smoke_recompute on
                            tiny Llama 3 models):
  smoke_test.py             oracle projector reproduces true logits
  smoke_fast.py             fast path == reference path
  smoke_recompute.py        recomputing every middle token == large-model prefill
  smoke_specprefill.py      SpecPrefill and drop-* configs
  smoke_multiq.py           multi-question configs
  smoke_rarecache.py        RaReCache map / fit / selection
  smoke_cross.py            cross-tokenizer path (different vocabularies; real Qwen3 <-> Llama 3 tables)
  smoke_r5.py               in_norm (reference vs folded path) and the stage-2 attention loss
  smoke_deep.py             deep handoff limits
  gpu_check_lah_graph.py    (GPU) graphed SpecPrefill look-ahead == eager
```
Models, data, self-distillation stores and checkpoints live under `$PREFILL_ROOT`, by default
`/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM`. The subdirectories are `models/`, `data/` (Qwen tokenizer),
`data_llama/` (Llama tokenizer), `checkpoints/` and `hf_cache/`. Evaluation outputs (`summary.md`, `results.json`,
`generations.jsonl`, `figures/`) go to `results/<run>/` in this directory, and SLURM logs go to `logs/`. The appendix of
`REPORT_2026-11-02_32B_llama.md` says which report covers each results directory.

## Environment
| Variable | Meaning | Default |
|---|---|---|
| `PREFILL_ROOT` | scratch root | `/scratch/drjieliu_root/drjieliu/shengkai/prefillLLM` |
| `PREFILL_SMALL_DIR` / `PREFILL_LARGE_DIR` | model pair | `$PREFILL_ROOT/models/Qwen3-0.6B` / `Qwen3-8B` |
| `PREFILL_DATA_DIR` | token pools of the **large** model's tokenizer | `$PREFILL_ROOT/data` (Llama targets: `data_llama`) |
| `PREFILL_LARGE_MAXMEM` | split the large model over GPUs, e.g. `"24GiB,76GiB"` (32B training) | unset = one GPU |

The slurm scripts set these variables themselves. Cross-tokenizer alignment is enabled automatically when the two
tokenizers differ. Its lookup tables are cached in `$PREFILL_ROOT/hf_cache/tokmap_*.npy`.

## Run
Every slurm script runs its steps sequentially in one job and can restart from a step (step names are in its header):
```bash
sbatch slurm/run_round2.slurm                                     # 0.6B -> 8B: pools -> self-distillation -> stage 2 -> eval
sbatch slurm/run_r6_32b_norm.slurm                                # 1.7B -> 32B training (2 H100)
sbatch slurm/eval_32b_suite.slurm                                 # 1.7B -> 32B full benchmark (1 H100)
sbatch slurm/prep_llama.slurm; sbatch slurm/llama_gen.slurm       # Llama pair: pools (CPU), self-distillation (2 L40S)
sbatch slurm/run_llama.slurm                                      # Llama pair: smoke, train, full benchmark (1 L40S)
sbatch -J pf_xq17ll8 --export=ALL,PAIR=q17_ll8 slurm/run_cross.slurm   # Qwen3-1.7B -> Llama-3.1-8B
sbatch -J pf_xll1q8  --export=ALL,PAIR=ll1_q8  slurm/run_cross.slurm   # Llama-3.2-1B -> Qwen3-8B
sbatch --export=ALL,START_AT=stage2 slurm/run_llama.slurm         # restart from a step
```
The "full benchmark" consists of eval_v2, main (RULER / QA / NIAH / GSM8K / code), speed, long (16k / 32k), multiq,
LongBench and RaReCache. 8B pairs fit on one GPU (`drjieliu-l40s` or `drjieliu-h100`); the stage-2 attention loss
needs an H100. 32B training needs 2 H100s, and 32B evaluation fits on one. Run the CPU tests with
`cd tests && python <name>.py`.

## Config strings
Every eval script takes `--configs` (comma-separated):

| String | Meaning |
|---|---|
| `big`, `small` | large / small model alone |
| `hyb` | projected cache, nothing recomputed |
| `<scorer>:<frac>[:<unit>]`, e.g. `attn_small:0.1:sent` | recompute `frac` of the middle tokens; the rest keep projected KV (recommended: `attn_small:0.1:sent`) |
| `drop-<scorer>:<frac>[:<unit>]` | same selection, the rest is **dropped** (ablation) |
| `specprefill:<frac>[:lahN]` | SpecPrefill baseline (official defaults, look-ahead 8) |
| `rarecache[-s][f]:<frac>` | RaReCache baseline (`-s`: force sink / tail recompute; needs `--rarecache_fit`) |

Scorers are `attn_small`, `attn_big`, `spec`, `random` and `recent`. Units are single tokens (no unit given),
`bN` (blocks of N tokens) and `sent` (sentences / lines). Multi-question configs (`ours:f`, `ours-acc:f`, `pre:f:unit[:g]`,
`spec-agn:f`, `spec-q1:f`, ...) are documented at the top of `prefill/multiq.py`.

## Using a trained projector
```python
from prefill.models import load_pair
from prefill.projector import Projector
from prefill.kv import hybrid_prefill, greedy_decode
tok, small, large = load_pair("cuda")                     # pair from PREFILL_SMALL_DIR / PREFILL_LARGE_DIR
proj, _ = Projector.load(".../checkpoints/r2_mix8k/projector.pt", "cuda"); proj.eval()
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
cache, logits = hybrid_prefill(large, small, proj, ids, n_sink=8, n_tail=64)   # TTFT ends here
out = greedy_decode(large, cache, logits, 256, eos_ids=(tok.eos_token_id,))
```
Fast path with selective recompute:
```python
from prefill.fast import FastQwen3, FoldedProjectorKV, FastHybrid, Graphed, kv_to_dynamic_cache
S, L = FastQwen3(small), FastQwen3(large)          # re-points HF weights to fused views, no extra memory (Llama too)
fh = FastHybrid(S, L, FoldedProjectorKV(proj, L), n_sink=8, n_tail=64)
kbuf, vbuf, logits = fh.prefill(ids_1d)            # or Graphed(fh.prefill, ids_1d) for a fixed length
out = greedy_decode(large, kv_to_dynamic_cache(kbuf, vbuf), logits, 256, eos_ids=(tok.eos_token_id,))
```
`prefill.recompute.RecomputeHybrid` (selective recompute, SpecPrefill) and `prefill.multiq.MultiQ` (shared document,
several questions) build on the same objects. `scripts/eval_recompute.py` and `scripts/eval_multiq.py` show the setup.

## Training recipe
- **Stage 1:** representation alignment, 8000 steps.
- **Stage 2:** end-to-end KL distillation on self-distilled responses of the large model, mix8k (4k → 8k length
  curriculum), 12000 steps. Round 5 onwards adds `in_norm`; it is required for 1.7B small models and harmless for 0.6B.
- **Data from round 2 onwards:**
  - Web 25%.
  - Extractive QA 20% (SQuAD v2, HotpotQA, TriviaQA).
  - Code 15% (codeparrot).
  - Chat 15% (UltraChat).
  - Math 10% (MetaMathQA few-shot).
  - Synthetic retrieval 15% (records, JSON uuid key-value, verbatim sentence copy; the templates differ from NIAH / RULER).
- **Test data:** held-out material only (validation sets, held-out repos, hash-split MetaMath), plus RULER 4k/8k
  (`simonjegou/ruler`), GSM8K, LongBench (`THUDM/LongBench`) and 16k / 32k QA / NIAH pools.

## Notes / deviations from the plan
* **SlimPajama:** `cerebras/SlimPajama-627B` is no longer reachable on the Hub (401), so the public subsample
  `DKYoon/SlimPajama-6B` is used instead.
* **Llama models:** they come from the unsloth mirrors, because the official meta-llama repos are gated and there is no
  HF token here. The Llama token pools are rebuilt with the same scripts, sizes and seeds under `data_llama/`.
* **Hidden states:** they are computed online, not cached, because one 4k sequence of 8B hidden states is 1.2 GB.
* **1.7B→32B training data:** it reuses the self-distillation responses generated by 8B. The teacher logits come from 32B.
* **Cross-family pairs:** the large model keeps its own tokens. Each position of the small model's sequence gets one
  small-vocabulary token, using C2C's "longest" strategy. On our English pools 97.8–100% of tokens map to an identical
  byte string (`results/tokalign/summary.md`). The cross-family runs have no "small alone" config.
* **LongBench code completion on Qwen3-32B:** raw prompts make the model answer in prose, so for this model the
  assistant turn is pre-filled with a code fence. These scores are comparable only within the 32B group.
* **RaReCache:** this is a reimplementation, because no code was released. With the paper's setting (no forced sink),
  position 0 gets a bad V map. `rarecache-s` forces sink / tail recompute and is the variant reported.
* **Resuming stage 2:** `train_stage2.py` does not save optimizer state. `--start_step` continues the LR schedule and
  curriculum with a short re-warmup. Do not re-warm an already trained projector to a high LR: that destroys learned
  retrieval.
* **Not implemented:** anchor layers, LoRA on the large model, and vLLM/SGLang integration.
