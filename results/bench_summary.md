# 长文本基准汇总：大模型单独 vs 我们 vs SpecPrefill

数据来自 `results/bench_<pair>/results.json`（scripts/eval_bench.py）和 `results/quality_<pair>/results.json`。F1 任务：贪心生成第一行与参考答案的 token F1；acc 任务：首 token 字母 logits 判分。我们 = 小模型注意力按句子选 10% 中间 token 重算，其余用投影 KV；SpecPrefill = 官方方法保留整段 prompt 的 10% / 20%。各 pair 的 prompt 上限、题数见各自的 summary.md。

数据来源与和原始表格的差异（scripts/prepare_bench.py）：
- LastingBench：github Seriousss/LastingBench 的 hotpotqa_antifact.jsonl（反事实改写的 HotpotQA，LongBench 格式），前 150 题。
- MuSiQue：LongBench 的 musique.jsonl 前 120 题。Qasper：allenai/qasper test 集，只取第一位标注者给出片段或是/否答案的问题，前 304 题。
- LooGLE：bigai-nlco/LooGLE 的 shortdep_qa（每篇 ≤2 题）和 shortdep_cloze（每篇 1 题）各 150 题；LooGLE-long 只取 ≥48k token 的文档（符合的只有 3 篇问答 + 16 篇完形文档，所以以完形为主）。
- CLIPPER：chtmp223/CLIPPER test 集按 id 取前 504 行，奇偶行交替问该行的真/假断言（各半）；整本书 56k–125k token，统一截断到 prompt 上限。
- LongHealth：github kbressem/LongHealth benchmark_v5.json，fold 0 = patient_01–04 的 80 题（5 选 1）。ContractNLI：kiddothe2b/contract-nli 的整份合同版（contract_nli_long）test 集，seed 0 抽 300 个（合同，假设）对，3 分类。
- QuALITY：全部 2086 题（results/quality_<pair>），不是原表的 1057 题。Shakespeare-CF 没有找到公开数据，未包含。
- 原表的 “clean” 子集筛选规则未知，这里没有做筛选；prompt 超过上限（8B 对 64k，32B 对 16k）从中间截断（LongBench 规则）。

## Qwen3-0.6B → Qwen3-8B（`06`）


| 任务 | 指标 | 题数 | 大模型单独 | 我们（重算 10%） | SpecPrefill 10% | SpecPrefill 20% | 我们 − 大模型 | 我们 − Spec 10% |
|---|---|---|---|---|---|---|---|---|
| QuALITY | acc | 2086 | 0.756 | 0.708 | 0.540 | 0.595 | -0.048 | +0.168 |

## Qwen3-1.7B → Qwen3-8B（`17`）


| 任务 | 指标 | 题数 | 大模型单独 | 我们（重算 10%） | SpecPrefill 10% | SpecPrefill 20% | 我们 − 大模型 | 我们 − Spec 10% |
|---|---|---|---|---|---|---|---|---|
| QuALITY | acc | 2086 | 0.756 | 0.726 | 0.602 | 0.651 | -0.030 | +0.124 |

## Qwen3-1.7B → Qwen3-32B（`32`）


| 任务 | 指标 | 题数 | 大模型单独 | 我们（重算 10%） | SpecPrefill 10% | SpecPrefill 20% | 我们 − 大模型 | 我们 − Spec 10% |
|---|---|---|---|---|---|---|---|---|
| QuALITY | acc | 2086 | 0.849 | 0.796 | 0.645 | 0.725 | -0.052 | +0.151 |

## Llama-3.2-1B → Llama-3.1-8B（`ll`）


| 任务 | 指标 | 题数 | 大模型单独 | 我们（重算 10%） | SpecPrefill 10% | SpecPrefill 20% | 我们 − 大模型 | 我们 − Spec 10% |
|---|---|---|---|---|---|---|---|---|
| QuALITY | acc | 2086 | 0.732 | 0.646 | 0.541 | 0.608 | -0.085 | +0.105 |

## 各 pair 对比（我们 / SpecPrefill 10% / 大模型）

| 任务 | Qwen3-0.6B → Qwen3-8B | Qwen3-1.7B → Qwen3-8B | Qwen3-1.7B → Qwen3-32B | Llama-3.2-1B → Llama-3.1-8B | Qwen3-1.7B → Llama-3.1-8B | Llama-3.2-1B → Qwen3-8B |
|---|---|---|---|---|---|---|
| LastingBench（F1） | - | - | - | - | - | - |
| MuSiQue（F1） | - | - | - | - | - | - |
| LooGLE（F1） | - | - | - | - | - | - |
| LooGLE-long（F1） | - | - | - | - | - | - |
| Qasper（F1） | - | - | - | - | - | - |
| CLIPPER（acc） | - | - | - | - | - | - |
| QuALITY（acc） | 0.708 / 0.540 / 0.756 | 0.726 / 0.602 / 0.756 | 0.796 / 0.645 / 0.849 | 0.646 / 0.541 / 0.732 | - | - |
| LongHealth（acc） | - | - | - | - | - | - |
| ContractNLI（acc） | - | - | - | - | - | - |
