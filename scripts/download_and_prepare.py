#!/usr/bin/env python
"""
Step 1 (CPU only): download Qwen3-0.6B / Qwen3-8B, stream FineWeb-Edu + SlimPajama, tokenise, and
save train/test token stores plus GSM8K under the scratch root. Idempotent: existing outputs are skipped.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prefill import config as C  # noqa: E402

os.environ.setdefault("HF_HOME", C.HF_CACHE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prefill.data import save_store  # noqa: E402


def download_models():
    for repo, local in ((C.SMALL_REPO, C.SMALL_DIR), (C.LARGE_REPO, C.LARGE_DIR)):
        if os.path.exists(os.path.join(local, "config.json")) and any(f.endswith(".safetensors") for f in os.listdir(local)):
            print(f"[models] {local} exists, skip")
            continue
        print(f"[models] downloading {repo} -> {local}", flush=True)
        snapshot_download(repo_id=repo, local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
        print(f"[models] done {repo}", flush=True)


def prepare_text(name, tok, n_train, n_test, min_tokens, max_tokens, batch=64):
    out_dir = os.path.join(C.DATA_DIR, name)
    if os.path.exists(os.path.join(out_dir, "test_offsets.npy")):
        print(f"[data] {name} exists, skip")
        return
    repo, cfg, field = C.DATASETS[name]
    print(f"[data] streaming {repo} ({cfg})", flush=True)
    ds = load_dataset(repo, cfg, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=5000)
    seqs, buf, seen, t0 = [], [], 0, time.time()
    need = n_train + n_test
    for ex in ds:
        buf.append(ex[field])
        if len(buf) < batch:
            continue
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        buf = []
        seen += batch
        for ids in enc:
            if len(ids) >= min_tokens:
                seqs.append(ids[:max_tokens])
        if seen % (batch * 50) == 0:
            print(f"[data] {name}: scanned {seen} docs, kept {len(seqs)}/{need} ({time.time()-t0:.0f}s)", flush=True)
        if len(seqs) >= need:
            break
    save_store(name, "test", seqs[:n_test])
    save_store(name, "train", seqs[n_test:need])


def prepare_gsm8k():
    if os.path.exists(os.path.join(C.GSM8K_DIR, "dataset_dict.json")):
        print("[data] gsm8k exists, skip")
        return
    ds = load_dataset("openai/gsm8k", "main")
    ds.save_to_disk(C.GSM8K_DIR)
    print("[data] saved gsm8k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=12000, help="sequences per source")
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--min_tokens", type=int, default=768)
    ap.add_argument("--max_tokens", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--sources", default="fineweb_edu,slimpajama")
    ap.add_argument("--skip_models", action="store_true")
    a = ap.parse_args()
    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    if not a.skip_models:
        download_models()
    tok = AutoTokenizer.from_pretrained(C.LARGE_DIR if os.path.exists(C.LARGE_DIR) else C.LARGE_REPO)
    for name in a.sources.split(","):
        prepare_text(name, tok, a.n_train, a.n_test, a.min_tokens, a.max_tokens)
    prepare_gsm8k()
    print("[done] all data ready under", C.SCRATCH_ROOT)


if __name__ == "__main__":
    main()
