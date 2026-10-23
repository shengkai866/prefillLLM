#!/usr/bin/env python
"""Download Hugging Face model repos into MODELS_DIR/<name> (weights, configs, tokenizer) and verify every shard
listed in the safetensors index is present. Re-running skips complete models and resumes partial downloads.

  python scripts/download_models.py Qwen/Qwen3-1.7B Qwen/Qwen3-32B
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from huggingface_hub import snapshot_download  # noqa: E402

from prefill import config as C  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"]


def missing_shards(local):
    """Shards named in the safetensors index (or the single model.safetensors) that are not on disk."""
    idx = os.path.join(local, "model.safetensors.index.json")
    if os.path.exists(idx):
        shards = set(json.load(open(idx))["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    return sorted(s for s in shards if not os.path.exists(os.path.join(local, s)))


def complete(local):
    return os.path.exists(os.path.join(local, "config.json")) and not missing_shards(local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for repo in a.repos:
        local = os.path.join(C.MODELS_DIR, repo.split("/")[-1])
        if complete(local):
            print(f"[models] {local} complete, skip", flush=True)
            continue
        for attempt in range(1, a.retries + 1):
            t0 = time.time()
            print(f"[models] {repo} -> {local} (attempt {attempt})", flush=True)
            try:
                snapshot_download(repo_id=repo, local_dir=local, allow_patterns=PATTERNS, max_workers=a.workers)
            except Exception as e:  # network stalls / rate limits: retry, the download resumes
                print(f"[models] {repo} attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(60 * attempt)
                continue
            if complete(local):
                size = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local) if f.endswith(".safetensors"))
                print(f"[models] done {repo}: {size / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)
                break
            print(f"[models] {repo} incomplete, missing {missing_shards(local)}", flush=True)
        else:
            sys.exit(f"[models] {repo} failed after {a.retries} attempts")


if __name__ == "__main__":
    main()
