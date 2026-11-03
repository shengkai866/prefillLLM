#!/usr/bin/env python
"""Build the C2C token lookup tables for both cross-family pairs (prefill/tokalign.py) and measure how lossy they are
on real prompts of the large model's pools: exact-token rate, bytes kept, and the small model's own tokenization length
for comparison. CPU only. Writes results/tokalign/summary.md and the cached tables under $PREFILL_ROOT/hf_cache."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prefill import config as C  # noqa: E402
from prefill.tokalign import build_table, coverage, token_bytes  # noqa: E402

PAIRS = [("Qwen3-1.7B", "Llama-3.1-8B-Instruct", "data_llama"), ("Llama-3.2-1B-Instruct", "Qwen3-8B", "data")]
POOLS = ["mix4k_test", "qa_eval", "gsm8k_eval", "code_eval"]


def pool_ids(tree, name, n=50):
    d = os.path.join(C.SCRATCH_ROOT, tree, "pools", name)
    if not os.path.exists(os.path.join(d, "offsets.npy")):
        return []
    toks, offs = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r"), np.load(os.path.join(d, "offsets.npy"))
    return [np.array(toks[offs[i]:offs[i + 1]], dtype=np.int64) for i in range(min(n, len(offs) - 1))]


def main():
    out_dir = os.path.join(C.RESULTS_DIR, "tokalign")
    os.makedirs(out_dir, exist_ok=True)
    L = ["# 跨分词器的 token 对齐（Cache-to-Cache 方法）", "",
         "大模型保留自己的 token；小模型的每个位置是对应大 token 的字符串用小模型分词器重新编码后最长的那个 token。"
         "“精确”= 小词表里有同样字节串的 token；“保留字节”= 小模型序列保留的大 token 字节占比；"
         "“小模型自然长度”= 同一文本用小模型分词器直接编码的长度（相对大模型序列）。", ""]
    res = {}
    for small, large, tree in PAIRS:
        ts = AutoTokenizer.from_pretrained(os.path.join(C.MODELS_DIR, small))
        tl = AutoTokenizer.from_pretrained(os.path.join(C.MODELS_DIR, large))
        table, stats, how = build_table(tl, ts)
        bl, bs = token_bytes(tl), token_bytes(ts)
        L += [f"## {small} ← {large} 的 prompt", "", f"词表映射：{stats}", "",
              "| 池 | 序列数 | 精确 token 占比 | 保留字节占比 | 小模型自然长度 / 大模型长度 |", "|---|---|---|---|---|"]
        res[f"{small}<-{large}"] = {"vocab": stats, "pools": {}}
        for pool in POOLS:
            seqs = pool_ids(tree, pool)
            if not seqs:
                continue
            same, kept, ratio = [], [], []
            for ids in seqs:
                a, b = coverage(ids, table, bl, bs)
                same.append(a)
                kept.append(b)
                text = tl.decode(ids, skip_special_tokens=False)
                ratio.append(len(ts(text, add_special_tokens=False)["input_ids"]) / len(ids))
            row = {"n": len(seqs), "exact": float(np.mean(same)), "bytes_kept": float(np.mean(kept)), "len_ratio": float(np.mean(ratio))}
            res[f"{small}<-{large}"]["pools"][pool] = row
            L.append(f"| {pool} | {row['n']} | {row['exact']:.3f} | {row['bytes_kept']:.3f} | {row['len_ratio']:.3f} |")
        # examples of lossy tokens
        lossy = [i for i in range(len(tl)) if how[i] == 3][:2000]
        ex = [(tl.decode([i]), ts.decode([int(table[i])])) for i in lossy[::max(1, len(lossy) // 12)]]
        L += ["", "多 token 取最长的例子（大 token → 小 token）：" + "；".join(f"`{a!r}`→`{b!r}`" for a, b in ex), ""]
        print(small, "<-", large, stats, res[f"{small}<-{large}"]["pools"], flush=True)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(L) + "\n")
    print("[done]", out_dir)


if __name__ == "__main__":
    main()
