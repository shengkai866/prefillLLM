"""Pre-tokenised datasets stored as flat uint32 token arrays + int64 offsets."""
import os
import random

import numpy as np
import torch

from .config import DATA_DIR


class TokenStore:
    def __init__(self, name, split):
        d = os.path.join(DATA_DIR, name)
        self.tokens = np.load(os.path.join(d, f"{split}_tokens.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(d, f"{split}_offsets.npy"))
        self.name = name

    def __len__(self):
        return len(self.offsets) - 1

    def length(self, i):
        return int(self.offsets[i + 1] - self.offsets[i])

    def get(self, i, max_len=None):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        if max_len is not None:
            b = min(b, a + max_len)
        return torch.from_numpy(np.array(self.tokens[a:b], dtype=np.int64))

    def indices_with_min_len(self, min_len):
        lens = np.diff(self.offsets)
        return np.nonzero(lens >= min_len)[0]


class MixedSampler:
    """Uniformly samples sequences across several TokenStores (sequence length >= min_len)."""

    def __init__(self, names, split, min_len, seed=0):
        self.stores = [TokenStore(n, split) for n in names]
        self.pools = [s.indices_with_min_len(min_len) for s in self.stores]
        self.lens = [np.diff(s.offsets) for s in self.stores]
        self.rng = random.Random(seed)
        for s, p in zip(self.stores, self.pools):
            print(f"[data] {s.name}/{split}: {len(s)} seqs, {len(p)} with len>={min_len}")

    def max_len(self):
        return int(max(l.max() for l in self.lens))

    def sample(self, max_len, at_least=None):
        """Random sequence truncated to max_len; with at_least=L only sequences of length >= L are drawn."""
        for _ in range(1000):
            k = self.rng.randrange(len(self.stores))
            pool = self.pools[k]
            i = int(pool[self.rng.randrange(len(pool))])
            if at_least is None or self.lens[k][i] >= at_least:
                return self.stores[k].get(i, max_len)
        raise RuntimeError(f"no sequence of length >= {at_least}")


def save_store(name, split, seqs):
    d = os.path.join(DATA_DIR, name)
    os.makedirs(d, exist_ok=True)
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate([np.asarray(s, dtype=np.uint32) for s in seqs]) if seqs else np.zeros(0, np.uint32)
    np.save(os.path.join(d, f"{split}_tokens.npy"), flat)
    np.save(os.path.join(d, f"{split}_offsets.npy"), offsets)
    print(f"[data] saved {name}/{split}: {len(seqs)} seqs, {flat.size} tokens -> {d}")
