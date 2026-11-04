# 首 token 延迟汇总（所有方法 CUDA graph，batch 1）

生成：2026-11-04 02:24:03。毫秒，括号内是相对同一次运行的大模型单独的加速比。“原始 fold”= `FoldedProjectorKV`（`PREFILL_FOLD=eager`），“融合 kernel”= `FusedFoldKV`（`prefill/fold_fused.py`，默认）；两者只在 projector → KV 阶段不同，大模型 / 小模型 / SpecPrefill 不经过这一阶段。数据：`results/fused_<tag>_speed`（融合）和 `results/fused_<tag>_speed_eagerfold`（原始），由 `slurm/rerun_fused_all.slurm` 产生。
