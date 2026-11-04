# 2026-11-03：projector → KV 的融合 kernel（fold-kernel 分支）

对应 `REPORT_2026-10-28_followup.md` 第 6.1 节第 1 点："projector fold 60 → 15–20 ms"。代码在分支 `fold-kernel`（worktree `prefillLLM-fold`），主目录未动。

## 结论

- **fold 阶段（projector + 大模型 K/V 生成）从 59.7 / 122 / 246 / 491 ms（4k / 8k / 16k / 32k，L40S，CUDA graph）压到 21.7 / 42.7 / 83.8 / 168 ms**，2.75–2.93 倍；1.7B 为 73 → 34、147 → 68、296 → 136、595 → 274 ms。数值与原实现一致（graph 下首 token logits 相对差 0），kernel 本身比原实现更接近 fp64。
- **0.6B → 8B 的首 token（重算 10%，推荐配置）：4k 172 → 137 ms，比 SpecPrefill 10%（158 ms）快 13%；8k 403 → 325（SpecPrefill 313，差 4%）；16k 1007 → 853（723）；32k 2741 → 2400（1800）。** 相对 8B 的加速从 2.33 / 2.32 / 2.08 / 1.85 倍变为 2.95 / 2.90 / 2.50 / 2.14 倍。"不重算"在所有长度都比 SpecPrefill 10% 快 25–35%（3.87 / 4.35 / 3.92 / 3.56 倍）。
- 16k 以上的差距不再在 fold 上（32k 时 fold 只占 2400 ms 中的 168），而在重算调用（重算的 token 对整段 T 做注意力）和小模型自己的 prefill，即 6.1 第 2 点。
- 下一个可以拿回来的是 fold 里的激活计算（`a_l = silu(W1·mix_l(h) + b1)`，约 10 ms @ 4k，和 kernel 本身一样长）；已写成第二个 Triton kernel（见第 5 节），单测排队中。

## 1. 原实现慢在哪

`FoldedProjectorKV`（`prefill/fast.py`）把 36 层的 projector 尾部 + input_layernorm + k/v 投影折成一次 `baddbmm`：`Y = a_l @ Wcat_l + b_l`，列为 `[K | V | norm | pad]`（8B：1024 + 1024 + 2049 → 4224）。层间本来就是 batch 维，并行度不缺；慢在 GEMM 之后：

- `Y` 是 `[36, 4096, 4224]` bf16（1.25 GB @ 4k）先写出去；
- 然后沿最后一维切片做 `.float()`、`pow(2)`、`sum`、`rsqrt`、乘、`.to(bf16)`、k_norm、`transpose`、RoPE、拷入 `kdst`，每步一个 kernel、整张读写一遍，而且切片是跨步访问。

4k 时 GEMM 的 FLOP 是 36 × 2 × 2048 × 4224 × 4096 = 2.55 TFLOP，L40S 下限约 13 ms；实测 60 ms，差的都是 epilogue。

## 2. 融合 kernel（`prefill/fold_fused.py`）

一个 Triton program 负责一层的 BM 个 token、整行：

1. 先循环 norm 列的 tile，在寄存器里累加每行的平方和，得到 RMSNorm 的 `inv = rsqrt(ss / d_l + eps)`（norm 列是折叠映射的 QR 因子，平方和就是 `||x_l||²`，和原实现相同）；
2. 再逐 head 算 K：每个 head 分 lo / hi 两个半 tile（64 列）累加，这样 k_norm（head 内 RMS）和 RoPE（`rotate_half` 配对 lo/hi）都在 tile 内完成，fp32 算完直接按 stride 写到 `kbuf[l, h, s:s+m, :]`；
3. 然后逐 head 算 V：乘 `inv` 直接写出。

中间张量完全不落地；全程 fp32，少了原实现里三处 bf16 舍入。变体 2 把 norm 阶段的 BN 和 head 阶段"一次两个 head"（4 个半 tile）作为 autotune 维度，少读一半 `a_l`；L40S 上比变体 1 快 1–2%。autotune 只做一次（`warm()` 在 4k 上调），之后的调用（含 CUDA graph 捕获）不再编译。

`prefill/fast.py` 新增 `build_fold(projector, large_fast)`：`PREFILL_FOLD=fused`（默认）| `eager`，CPU 上自动 eager；`eval_recompute.py`、`bench_speed.py` 走它，其它脚本通过 `stage_fold` 自然用上。

## 3. kernel 单测（`tests/test_fold_fused.py`，随机 8B 形状权重）

| | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| H100：eager → fused | 20.0 → 11.7 ms | 40.3 → 23.0 | 80.5 → 46.3 | 161 → 92.5 |
| L40S：eager → fused（v2） | 64.4 → 25.4 | 129 → 49.1 | 258 → 99.0 | 516 → 198 |
| L40S fused 拆分：激活 + kernel | 13.4 + 11.9 | 26.8 + 22.3 | 51.9 + 47.0 | 103 + 95.3 |

- kernel 本身在 L40S 已经到 GEMM 的地板（11.9 ms ≈ 2.55 TFLOP / 215 TFLOPs）。
- 精度（对同一 `a_l` 的 fp64 参考）：eager 的 K 相对误差 3.2e-3、V 1.9e-3；fused 均为 1.4e-3（只剩最终一次 bf16 舍入）。Llama 式（无 k_norm）相同。
- **激活计算（`rms_unit` → `W1` → 28 层加权混合 → +b1 → silu）占了另一半**：K=28 的瘦 matmul 加两次逐元素遍历 `[36, n, 2048]`，接近 kernel 本身的时间。

## 4. 全流程首 token（`results/speed_fold_{eager,fused}_{06,17}`，L40S，所有方法 CUDA graph）

0.6B → 8B，毫秒（括号内相对 8B 单独）：

| 配置 | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| 8B 单独 | 400 / 404 | 934 / 942 | 2092 / 2129 | 5066 / 5126 |
| 混合，不重算：eager → fused | 142 → **105**（2.83 → 3.87） | 294 → **217**（3.17 → 4.35） | 700 → **543**（2.99 → 3.92） | 1763 → **1440**（2.87 → 3.56） |
| 重算 10%：eager → fused | 172 → **137**（2.33 → 2.95） | 403 → **325**（2.32 → 2.90） | 1007 → **853**（2.08 → 2.50） | 2741 → **2400**（1.85 → 2.14） |
| 重算 20%：eager → fused | 218 → 183（1.83 → 2.21） | 507 → 432（1.84 → 2.18） | 1317 → 1162（1.59 → 1.83） | 3803 → 3468（1.33 → 1.48） |
| 丢弃其余 20% | 139 / 139 | 304 / 305 | 750 / 755 | 1987 / 1976 |
| SpecPrefill 10% | 158（2.55） | 313（3.00） | 723（2.92） | 1800（2.82） |
| SpecPrefill 20% | 194（2.08） | 384（2.44） | 881（2.40） | 2230（2.28） |
| SpecPrefill 30% | 230（1.76） | 465（2.02） | 1089（1.94） | 2682（1.90） |

1.7B → 8B：

| 配置 | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| 混合，不重算：eager → fused | 200 → **163**（1.98 → 2.49） | 419 → **352**（2.18 → 2.69） | 975 → **832**（2.19 → 2.59） | 2291 → **1968**（2.24 → 2.59） |
| 重算 10%：eager → fused | 228 → **195**（1.73 → 2.08） | 526 → **461**（1.74 → 2.05） | 1285 → **1138**（1.66 → 1.90） | 3260 → **2959**（1.58 → 1.72） |
| SpecPrefill 10% | 223 / 227（1.78） | 438 / 442（2.12） | 971 / 977（2.21） | 2259 / 2245（2.27） |

分阶段（`results/speed_fold_stages_*`，graph 内）：

| 阶段 | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| 0.6B prefill | 49 | 136 | 412 | 1197 |
| fold：eager → fused | 59.7 → **21.7** | 122 → **42.7** | 246 → **83.8** | 491 → **168** |
| 8B 开头 + 尾部（72 token） | 35 | 41 | 53 | 76 |
| 1.7B prefill | 96 | 241 | 644 | 1624 |
| fold（1.7B）：eager → fused | 73 → **34** | 147 → **68** | 296 → **136** | 595 → **274** |

- 短上下文（4k–8k）达到了 6.1 的目标：重算 10% 与 SpecPrefill 10% 同速或更快，可以在同速度下比质量了。
- 16k / 32k 还差 18% / 33%。fold 已经是小头；剩下是重算调用（32k 时约 1000 ms：3290 个重算 token 对整段 32k 做注意力，36 层）和 0.6B 自己的 prefill（1197 ms）。
- fused 的 `drop`、SpecPrefill、8B 单独不受影响（它们不走 fold），数字的小差异是运行间噪声（≤ 2%）。

## 5. 下一步

1. **激活计算融合**（已写，`_mix_silu_kernel`，commit d62d6ae，单测排队中）：把 28 层加权混合 + 偏置 + silu 做成对 `H1` 的一次遍历（每个元素只读一次），预期 10 ms → 1–2 ms @ 4k，fold 整体到 ~14 ms，重算 10% 的 4k 首 token 约 130 ms。
2. **重算调用的注意力范围**（6.1 第 2 点）：重算 token 只对"开头 + 本句前文窗口 + 已重算位置"做注意力，用变长 flash 接口一次算完。这是 16k 以上追 SpecPrefill 的唯一途径。
3. 合并到 main：等主目录上正在跑的 quality 任务（`pf_qual*`，顺序起多个 python 进程、每步重新 import）结束后再合并，避免中途切换实现。
