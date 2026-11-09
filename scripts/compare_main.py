"""
Side-by-side of the overview table (## 总表) of two eval_recompute.py result dirs, matched by config label, e.g. the same
configs under PREFILL_ATTN=sdpa (results/fused_<tag>_main) and triton (results/attn_verify_<tag>):
  python scripts/compare_main.py results/fused_06_main results/attn_verify_06 [--out results/attn_verify_summary.md]
Appends to --out (one section per call).
"""
import argparse
import os


def overview(path):
    """{label: {column: value}} of the first '## 总表' table in <dir>/summary.md"""
    rows, hdr, on = {}, None, False
    for line in open(os.path.join(path, "summary.md")):
        if line.startswith("## "):
            if rows:  # the overview table is done
                break
            on = line.strip() == "## 总表"
            continue
        if not on or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if hdr is None:
            hdr = cells
        elif not set(line.strip()) <= set("|-"):
            rows[cells[0]] = dict(zip(hdr[1:], cells[1:]))
    return hdr or [], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    hr, ref = overview(a.ref)
    _, new = overview(a.new)
    cols = [c for c in hr[1:] if not c.startswith("实际重算比例")]
    L = [f"### `{a.ref}` → `{a.new}`", "", "| 配置 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for lab, r in new.items():
        if lab not in ref:
            continue
        cells = []
        for c in cols:
            x, y = ref[lab].get(c, "-"), r.get(c, "-")
            try:
                d = float(y) - float(x)
                cells.append(f"{x} → {y}（{d:+.3f}）")
            except ValueError:
                cells.append(f"{x} → {y}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        new_file = not os.path.exists(a.out)
        with open(a.out, "a") as f:
            if new_file:
                f.write("# 注意力 kernel 正确性：同一配置 SDPA（旧）→ Triton kernel（新）\n\n"
                        "左边是 `results/fused_<tag>_main`（PREFILL_ATTN=sdpa 时代），右边是合并后用默认 kernel 重跑（slurm/verify_attn.slurm）。"
                        "RULER / 问答各子集 15–50 例，差 0.02 以内视为噪声。\n\n")
            f.write(text + "\n")


if __name__ == "__main__":
    main()
