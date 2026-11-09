"""
Per-item scores from a run's generations.jsonl, for paired statistics across configs and runs
(scripts/paired_ci.py, scripts/summarize_mid.py).

load(run_dir) -> {bench: {item_key: {config: {"score": x, "contains": y or None, "prompt_len": n or None}}}}
generations.jsonl is appended to on every (re-)run, so the LAST record of an (item, config) wins.

Benchmarks (item key -> bench name):
  main       ruler@<len> (score), qa@4k / qa@8k (score = F1 of the first line; contains = a normalised gold answer
             occurs anywhere in the output), niah@<len>
  long       qa_long@16k / @32k, niah_long@<len>
  multiq     multiq@4k / @8k (F1 / contains, every question of every document)
  longbench  gov_report, qmsum, lcc, repobench-p (score)
  quality    quality (accuracy over all 2086 questions), quality_mq@4k / @8k (multi-question, accuracy)
  bench      lastingbench, musique, qasper, loogle, loogle_long, clipper, longhealth, contractnli (score);
             contains computed here for the free-form answers (not for LooGLE cloze items)
GSM8K, code completion and LongBench GSM8K 32-shot have no per-item log and are not covered.
"""
import json
import os
import re
import string

STEP_OF = {"main": ("niah", "ruler", "qa"), "long": ("niah_long", "qa_long"), "multiq": ("qa",), "quality": ("quality",),
           "quality_mq": ("qa",), "longbench": ("gov_report", "qmsum", "lcc", "repobench-p"),
           "bench": ("lastingbench", "musique", "qasper", "loogle", "loogle_long", "clipper", "longhealth", "contractnli")}


def norm_ans(s):  # same as scripts/evaluate_v2.py
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(output, answers):
    if not answers or not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
        return None
    return float(any(norm_ans(a) in norm_ans(output) for a in answers if norm_ans(a)))


def key_of(kind, g):
    """(bench, item key) of one generations.jsonl record, or None to skip."""
    t = g.get("task")
    if kind == "main":
        if t == "ruler":
            return f"ruler@{g['length']}", (g["sub"], g["idx"])
        if t == "niah":
            return f"niah@{g['length']}", g["idx"]
        if t == "qa":
            return f"qa@{g['bucket']}", (g["source"], g["idx"])
    elif kind == "long":
        if t == "niah_long":
            return f"niah_long@{g['length']}", g["idx"]
        if t == "qa_long":
            return f"qa_long@{g['bucket']}", (g["source"], g["idx"])
    elif kind == "multiq" and t == "qa":
        return f"multiq@{g['bucket']}", (g["doc"], g["q"])
    elif kind == "quality_mq" and t == "qa":
        return f"quality_mq@{g['bucket']}", (g["doc"], g["q"])
    elif kind == "quality" and t == "quality":
        return "quality", g["question_id"]
    elif kind in ("longbench", "bench") and t in STEP_OF[kind]:
        return t, g["id"]
    return None


def score_of(kind, g):
    if "score" in g:
        s = g["score"]
    else:
        s = g.get("f1")
    c = g.get("contains")
    if c is None and kind == "bench" and g.get("task") in ("lastingbench", "musique", "qasper", "loogle", "loogle_long") \
            and g.get("subset") != "cloze":
        c = contains(g.get("output", ""), g.get("answers"))
    if kind == "quality_mq":  # multiple choice: f1 = em = contains = accuracy
        c = None
    return float(s), (None if c is None else float(c))


def load(run_dir, kind):
    p = os.path.join(run_dir, "generations.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = key_of(kind, g)
        if k is None or g.get("config") is None:
            continue
        bench, item = k
        s, c = score_of(kind, g)
        out.setdefault(bench, {}).setdefault(json.dumps(item), {})[g["config"]] = \
            {"score": s, "contains": c, "prompt_len": g.get("prompt_len")}
    return out


def kind_of(run_name):
    """results/<run_name> -> loader kind ('main', 'bench', ...)."""
    for k in ("quality_mq", "quality", "longbench", "multiq", "main", "long", "bench"):
        if run_name.startswith(k + "_") or run_name.endswith("_" + k) or f"_{k}_" in run_name or run_name == k:
            return k
    raise ValueError(run_name)
