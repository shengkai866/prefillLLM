# 长文本基准：LastingBench / MuSiQue / LooGLE / Qasper / CLIPPER / LongHealth / ContractNLI

GPU：NVIDIA H100 80GB HBM3；生成：2026-11-04 00:12:05；小模型 Llama-3.2-1B-Instruct，大模型 Llama-3.1-8B-Instruct，projector 第 12000 步；prompt 上限 65536 token（超出的上下文从中间截断），Qwen3 用 YaRN ×3。
F1 任务贪心生成后取第一行与参考答案算 token F1（LooGLE 完形：每个 mask 的 F1 取平均）；acc 任务按首 token 的字母 logits 判分。问题/选项始终在大模型自己计算的尾部里。“重算比例”对我们的方法是中间 token 被大模型重算的比例，对 SpecPrefill 是整段 prompt 保留的比例。

## 总表（每个任务的整体分数）

| 配置 | LastingBench（多跳，F1） | MuSiQue（F1） | LooGLE（F1） | Qasper（F1） | LongHealth（acc） | ContractNLI（acc） |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 0.435 | 0.351 | 0.408 | 0.560 | 0.800 | 0.620 |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 0.382 | 0.319 | 0.333 | 0.526 | 0.750 | 0.533 |
| SpecPrefill 保留整段 10% | 0.386 | 0.291 | 0.404 | 0.334 | 0.400 | 0.477 |
| SpecPrefill 保留整段 20% | 0.382 | 0.347 | 0.399 | 0.450 | 0.512 | 0.483 |

题数与平均 prompt 长度：LastingBench 150 题（13069 token）；MuSiQue 120 题（15647 token）；LooGLE 300 题（28898 token）；Qasper 304 题（4757 token）；LongHealth 80 题（11931 token）；ContractNLI 300 题（2377 token）

## LastingBench（多跳，F1）

| 配置 | 重算比例 | 每题秒数 | 整体 |  |
|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 0.7 | 0.435 |  |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.1% | 0.4 | 0.382 |  |
| SpecPrefill 保留整段 10% | 10.0% | 0.3 | 0.386 |  |
| SpecPrefill 保留整段 20% | 20.0% | 0.3 | 0.382 |  |

题数：all 150

## MuSiQue（F1）

| 配置 | 重算比例 | 每题秒数 | 整体 |  |
|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 0.9 | 0.351 |  |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.1% | 0.5 | 0.319 |  |
| SpecPrefill 保留整段 10% | 10.0% | 0.3 | 0.291 |  |
| SpecPrefill 保留整段 20% | 20.0% | 0.4 | 0.347 |  |

题数：all 120

## LooGLE（F1）

| 配置 | 重算比例 | 每题秒数 | 整体 | 完形 | 问答 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 2.9 | 0.408 | 0.406 | 0.409 |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.1% | 2.4 | 0.333 | 0.284 | 0.382 |
| SpecPrefill 保留整段 10% | 10.0% | 1.0 | 0.404 | 0.379 | 0.429 |
| SpecPrefill 保留整段 20% | 20.0% | 1.2 | 0.399 | 0.392 | 0.406 |

题数：all 300，完形 150，问答 150

## Qasper（F1）

| 配置 | 重算比例 | 每题秒数 | 整体 | 片段 | 是/否 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 0.5 | 0.560 | 0.556 | 0.576 |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.4% | 0.4 | 0.526 | 0.513 | 0.581 |
| SpecPrefill 保留整段 10% | 10.0% | 0.2 | 0.334 | 0.356 | 0.242 |
| SpecPrefill 保留整段 20% | 20.0% | 0.3 | 0.450 | 0.474 | 0.351 |

题数：all 304，片段 245，是/否 59

## LongHealth（acc）

| 配置 | 重算比例 | 每题秒数 | 整体 | patient_01 | patient_02 | patient_03 | patient_04 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 0.5 | 0.800 | 0.800 | 0.750 | 0.800 | 0.850 |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.1% | 0.3 | 0.750 | 0.800 | 0.700 | 0.750 | 0.750 |
| SpecPrefill 保留整段 10% | 10.0% | 0.2 | 0.400 | 0.400 | 0.500 | 0.400 | 0.300 |
| SpecPrefill 保留整段 20% | 20.0% | 0.2 | 0.512 | 0.500 | 0.600 | 0.450 | 0.500 |

题数：all 80，patient_01 20，patient_02 20，patient_03 20，patient_04 20

## ContractNLI（acc）

| 配置 | 重算比例 | 每题秒数 | 整体 | 矛盾 | 蕴含 | 未提及 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct 单独 | 100.0% | 0.1 | 0.620 | 0.200 | 0.746 | 0.583 |
| Llama-3.2-1B-Instruct 注意力 10%，按句子 | 10.9% | 0.1 | 0.533 | 0.133 | 0.514 | 0.644 |
| SpecPrefill 保留整段 10% | 10.1% | 0.1 | 0.477 | 0.000 | 0.210 | 0.864 |
| SpecPrefill 保留整段 20% | 20.0% | 0.1 | 0.483 | 0.000 | 0.217 | 0.871 |

题数：all 300，矛盾 30，蕴含 138，未提及 132

