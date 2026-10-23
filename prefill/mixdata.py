"""
Round-2 data: prompt builders for a mixture of task types, and a prompt-pool storage format.

Every builder returns a dict  {"ids": [token ids of the full prompt], "type": ..., "source": ..., **eval_meta}
whose length is <= the requested target L (and close to it when the source allows).
Chat-style prompts are rendered with the Qwen3 chat template (thinking disabled, generation prompt appended),
raw-style prompts (web, code) are plain text continuations.

Types (train mixture):  web | qa | code | chat | math | retrieval
Train and test material always come from disjoint sources (official splits, or a hash split when there is none).
"""
import hashlib
import json
import os
import random
import re
import string
import uuid

import numpy as np

from .config import DATA_DIR

POOL_DIR = os.path.join(DATA_DIR, "pools")

QA_INSTR = "Answer the question based on the given documents. Only give the answer, without any other words."


def stable_hash(s, mod):
    return int(hashlib.md5(s.encode("utf-8", "ignore")).hexdigest(), 16) % mod


# ------------------------------------------------------------------------------------------------ pool storage
class PromptPool:
    """tokens.npy (uint32 flat) + offsets.npy + meta.jsonl (one json per prompt, without the ids)."""

    def __init__(self, name):
        d = os.path.join(POOL_DIR, name)
        self.dir = d
        self.tokens = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(d, "offsets.npy"))
        self.meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]

    def __len__(self):
        return len(self.meta)

    def ids(self, i):
        return np.array(self.tokens[self.offsets[i]:self.offsets[i + 1]], dtype=np.int64)

    @staticmethod
    def exists(name):
        return os.path.exists(os.path.join(POOL_DIR, name, "meta.jsonl"))

    @staticmethod
    def save(name, items):
        d = os.path.join(POOL_DIR, name)
        os.makedirs(d, exist_ok=True)
        lens = np.array([len(x["ids"]) for x in items], dtype=np.int64)
        np.save(os.path.join(d, "tokens.npy"), np.concatenate([np.asarray(x["ids"], dtype=np.uint32) for x in items]))
        np.save(os.path.join(d, "offsets.npy"), np.concatenate([[0], np.cumsum(lens)]))
        with open(os.path.join(d, "meta.jsonl"), "w") as f:
            for x in items:
                f.write(json.dumps({k: v for k, v in x.items() if k != "ids"}) + "\n")
        print(f"[pool] saved {name}: {len(items)} prompts, {lens.sum()} tokens, mean len {lens.mean():.0f}", flush=True)


# ------------------------------------------------------------------------------------------------ helpers
class Tok:
    def __init__(self, tok):
        self.tok = tok

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def n(self, text):
        return len(self.enc(text))

    def chat(self, messages):
        """Token ids of a chat prompt with the generation prompt appended (thinking disabled)."""
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.enc(text), text


def fill_docs(T, docs_first, distractors, budget, rng, sep="\n\n"):
    """docs_first: required doc texts; distractors: list of (text, n_tokens). Adds random distractors until
    ~budget tokens, shuffles the order. Returns text, or None if the required docs alone exceed the budget."""
    n = sum(T.n(d) + 6 for d in docs_first)
    if n > budget:
        return None
    docs = list(docs_first)
    for _ in range(600):
        if n >= budget - 30:
            break
        d, dn = rng.choice(distractors)
        if n + dn + 6 > budget:
            continue
        docs.append(d)
        n += dn + 6
    rng.shuffle(docs)
    return sep.join(f"Document [{i + 1}]: {d}" for i, d in enumerate(docs))


def crop_text_to_tokens(T, text, budget, must_contain=None, rng=None):
    """A contiguous slice of text with <= budget tokens that contains `must_contain` (if given)."""
    pos, alen = 0, 0
    if must_contain:
        pos = text.lower().find(must_contain.lower())
        if pos < 0:
            return None
        alen = len(must_contain)
    sample = text[:20000]
    cpt = len(sample) / max(1, T.n(sample))  # chars per token
    win = int(budget * cpt * 0.95)
    for _ in range(8):
        if win <= alen + 10:
            return None
        lo = pos - int((win - alen) * (rng.random() if rng else 0.5)) if must_contain else 0
        lo = max(0, min(lo, len(text) - win))
        piece = text[lo:lo + win]
        n = T.n(piece)
        if n <= budget:
            if must_contain and must_contain.lower() not in piece.lower():
                return None
            return piece
        win = int(win * budget / n * 0.97)
    return None


def chunk_text(text, max_chars=2400, min_chars=400):
    """Split a long text into paragraph groups of roughly 100-600 tokens (used as distractor documents)."""
    out, cur = [], ""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) > max_chars and len(cur) >= min_chars:
            out.append(cur)
            cur = ""
        cur = (cur + "\n" + para).strip()
        if len(cur) > max_chars:
            out.append(cur[:max_chars])
            cur = ""
    if len(cur) >= min_chars:
        out.append(cur)
    return out


# ------------------------------------------------------------------------------------------------ builders
class Builders:
    """Holds source material for one split ('train' or 'test') and builds prompts of a target length."""

    def __init__(self, tok, split, seed):
        self.T = Tok(tok)
        self.split = split
        self.rng = random.Random(seed)
        self.web_docs = []        # list of token-id lists (long enough for raw continuation prompts)
        self.web_texts = []       # plain texts for haystacks
        self.squad, self.squad_ctx = [], []      # examples, distractors as (text, n_tokens)
        self.hotpot, self.hotpot_par = [], []
        self.trivia, self.trivia_ctx = [], []
        self.code_files = []      # (header, content)
        self.chats = []           # list of message lists
        self.math = []            # (query, response)
        self.words = []

    # -- web (raw continuation)
    def web(self, L):
        cands = [d for d in self.rng.sample(self.web_docs, min(200, len(self.web_docs))) if len(d) >= L]
        if not cands:
            return None
        return {"ids": list(cands[0][:L]), "type": "web", "source": "web", "fmt": "raw"}

    # -- extractive QA (chat)
    def _qa_prompt(self, docs_text, question):
        return [{"role": "user", "content": f"{QA_INSTR}\n\n{docs_text}\n\nQuestion: {question}"}]

    def qa(self, L, kind=None):
        kind = kind or self.rng.choice(["squad", "hotpot", "trivia"])
        overhead = 60
        for _ in range(20):
            if kind == "squad":
                ex = self.rng.choice(self.squad)
                req, dis, q, ans = [ex["context"]], self.squad_ctx, ex["question"], ex["answers"]
            elif kind == "hotpot":
                ex = self.rng.choice(self.hotpot)
                req, dis, q, ans = ex["paragraphs"], self.hotpot_par, ex["question"], [ex["answer"]]
            else:
                ex = self.rng.choice(self.trivia)
                q, ans = ex["question"], ex["answers"]
                budget = L - overhead - self.T.n(q)
                # gold wiki text, cropped around an answer occurrence, uses ~60% of the budget at most
                hit = next((a for a in ans if a.lower() in ex["context"].lower()), None)
                if hit is None:
                    continue
                gold = crop_text_to_tokens(self.T, ex["context"], int(budget * self.rng.uniform(0.3, 0.6)), hit, self.rng)
                if gold is None:
                    continue
                req, dis = [gold], self.trivia_ctx
            budget = L - overhead - self.T.n(q)
            text = fill_docs(self.T, req, dis, budget, self.rng)
            if text is None:
                continue
            ids, _ = self.T.chat(self._qa_prompt(text, q))
            if len(ids) <= L:
                return {"ids": ids, "type": "qa", "source": kind, "fmt": "chat", "question": q, "answers": ans}
        return None

    # -- code (raw continuation); also next-line completion prompts for evaluation
    def _code_text(self, min_tokens):
        parts, n = [], 0
        while n < min_tokens:
            hdr, content = self.rng.choice(self.code_files)
            piece = f"# ===== {hdr} =====\n{content}\n"
            parts.append(piece)
            n += self.T.n(piece)
            if len(parts) > 60:
                break
        return "".join(parts)

    def code(self, L):
        ids = self.T.enc(self._code_text(L + 50))
        if len(ids) < L:
            return None
        return {"ids": ids[:L], "type": "code", "source": "codeparrot", "fmt": "raw"}

    def code_nextline(self, L):
        """Prompt ends at a line boundary close to L tokens; target is the next substantive line."""
        text = self._code_text(L + 400)
        lines = text.split("\n")
        prefix_n, cut = 0, None
        acc = []
        for i, ln in enumerate(lines[:-1]):
            acc.append(ln)
            prefix_n += self.T.n(ln + "\n")
            if prefix_n > L - 20:
                break
            nxt = lines[i + 1].strip()
            if prefix_n > 0.8 * L and len(nxt) >= 12 and not nxt.startswith(("#", '"""', "'''")):
                cut = i + 1
        if cut is None:
            return None
        prompt = "\n".join(lines[:cut]) + "\n"
        ids = self.T.enc(prompt)
        if len(ids) > L:
            return None
        return {"ids": ids, "type": "code_nextline", "source": "codeparrot", "fmt": "raw", "target": lines[cut].strip()}

    # -- chat (multi-conversation history, final user turn)
    def chat(self, L):
        for _ in range(20):
            final = self.rng.choice(self.chats)
            last_user = max(i for i, m in enumerate(final) if m["role"] == "user")
            tail = final[:last_user + 1]
            ids, _ = self.T.chat(tail)
            if len(ids) > L:
                continue
            hist, misses = [], 0
            for _ in range(60):
                conv = self.rng.choice(self.chats)
                cand = conv + hist
                cids, _ = self.T.chat(cand + tail)
                if len(cids) > L:
                    misses += 1
                    if misses >= 8:
                        break
                    continue
                hist, ids = cand, cids
                if len(ids) > 0.9 * L:
                    break
            return {"ids": ids, "type": "chat", "source": "ultrachat", "fmt": "chat"}
        return None

    # -- math (few-shot as chat turns)
    def math_fewshot(self, L):
        q, _ = self.rng.choice(self.math)
        msgs = [{"role": "user", "content": q}]
        ids, _ = self.T.chat(msgs)
        if len(ids) > L:
            return None
        misses = 0
        for _ in range(80):
            sq, sr = self.rng.choice(self.math)
            cand = [{"role": "user", "content": sq}, {"role": "assistant", "content": sr}] + msgs
            cids, _ = self.T.chat(cand)
            if len(cids) > L:
                misses += 1
                if misses >= 8:
                    break
                continue
            msgs, ids = cand, cids
            if len(ids) > 0.9 * L:
                break
        return {"ids": ids, "type": "math", "source": "metamath", "fmt": "chat"}

    # -- synthetic retrieval (templates differ from the NIAH / RULER test templates)
    def _haystack(self, budget):
        parts, n = [], 0
        while n < budget:
            t = self.rng.choice(self.web_texts)
            parts.append(t)
            n += self.T.n(t)
        ids = self.T.enc(" ".join(parts))[:budget]
        return self.T.tok.decode(ids)

    def _name(self):
        return f"{self.rng.choice(self.words)}-{self.rng.choice(self.words)}"

    def retrieval(self, L, kind=None):
        kind = kind or self.rng.choice(["records", "kv", "copy"])
        r = self.rng
        overhead = 120
        if kind == "records":
            hay = self._haystack(L - overhead - 60)
            k = r.randint(1, 5)
            recs = []
            for _ in range(k):
                code = "".join(r.choice(string.digits) for _ in range(r.randint(6, 9))) if r.random() < 0.6 else str(uuid.UUID(int=r.getrandbits(128)))[:13]
                recs.append((self._name(), code))
            sents = re.split(r"(?<=[.!?])\s+", hay)
            for name, code in recs:
                sents.insert(r.randint(0, len(sents)), f"Record: the access code assigned to {name} is {code}.")
            name, code = r.choice(recs)
            content = " ".join(sents) + f"\n\nWhat is the access code assigned to {name}? Reply with the code only."
            ans = [code]
        elif kind == "kv":
            items, n = {}, 0
            budget = L - overhead
            while n < budget:
                kk, vv = str(uuid.UUID(int=r.getrandbits(128))), str(uuid.UUID(int=r.getrandbits(128)))
                items[kk] = vv
                n += 34
            keys = list(items)
            while self.T.n(json.dumps(items)) > budget and len(keys) > 2:
                items.pop(keys.pop())
            key = r.choice(keys)
            content = f"JSON data:\n{json.dumps(items)}\n\nIn the JSON object above, what is the value associated with key \"{key}\"? Reply with the value only."
            ans = [items[key]]
        else:
            hay = self._haystack(L - overhead - 80)
            sents = [s for s in re.split(r"(?<=[.!?])\s+", hay) if len(s.split()) >= 5]
            if len(sents) < 4:
                return None
            i = r.randint(1, len(sents) - 2)
            content = hay + f"\n\nRepeat verbatim the sentence that comes immediately after this sentence in the text above: \"{sents[i]}\""
            ans = [sents[i + 1]]
        ids, _ = self.T.chat([{"role": "user", "content": content}])
        if len(ids) > L:
            return None
        return {"ids": ids, "type": "retrieval", "source": kind, "fmt": "chat", "answers": ans}

    def build(self, typ, L):
        fn = {"web": self.web, "qa": self.qa, "code": self.code, "chat": self.chat, "math": self.math_fewshot,
              "retrieval": self.retrieval}[typ]
        for _ in range(10):
            x = fn(L)
            if x is not None:
                return x
        return None


MIX = {"web": 0.25, "qa": 0.20, "code": 0.15, "chat": 0.15, "math": 0.10, "retrieval": 0.15}


def build_pool(B: Builders, n, lmin, lmax, mix=MIX, log_every=1000, strict=False):
    """n prompts with type proportions `mix` and target lengths uniform in [lmin, lmax].
    strict=True rejects prompts that came out shorter than lmin (used for the length-bucketed test set)."""
    types = []
    for t, p in mix.items():
        types += [t] * int(round(n * p))
    B.rng.shuffle(types)
    out, fails = [], 0
    for i, t in enumerate(types):
        x = None
        for _ in range(6 if strict else 1):
            x = B.build(t, B.rng.randint(lmin, lmax))
            if x is not None and (not strict or len(x["ids"]) >= lmin):
                break
            x = None
        if x is None:
            fails += 1
            continue
        out.append(x)
        if (i + 1) % log_every == 0:
            print(f"[pool] {i + 1}/{len(types)} built ({fails} failed)", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ NIAH (round-1 test template)
def passcode_niah(T: Tok, web_texts, L, rng):
    code = "".join(rng.choice("0123456789") for _ in range(6))
    needle = f"The secret passcode is {code}."
    words = []
    hay_ids = []
    while len(hay_ids) < L - 80:
        words.extend(rng.choice(web_texts).split())
        hay_ids = T.enc(" ".join(words))
    hay = T.tok.decode(hay_ids[: L - 80])
    cut = int(len(hay) * rng.uniform(0.05, 0.95))
    cut = hay.find(" ", cut) if hay.find(" ", cut) > 0 else cut
    prompt = hay[:cut] + " " + needle + " " + hay[cut:] + \
        "\n\nQuestion: What is the secret passcode mentioned in the text above? Reply with only the passcode."
    ids, _ = T.chat([{"role": "user", "content": prompt}])
    return {"ids": ids, "type": "niah", "source": "passcode", "fmt": "chat", "answers": [code]}
