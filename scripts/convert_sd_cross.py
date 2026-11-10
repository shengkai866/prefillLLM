#!/usr/bin/env python
"""
Cross-family REVERSE pair data (e.g. Llama-3.1-8B prefills -> Qwen3-1.7B decodes): the stage-2 sequences must be the
TEACHER's (the prefilling 8B's) answers, written in the DECODER's tokens and chat template. The teacher family's
self-distillation stores (sd_mix4k / sd_mixlong / sd_mixtest of its own data tree) are converted by text:
  * chat prompts: the messages are parsed out of the source chat template (Qwen3 <|im_start|>role ... <|im_end|>,
    Llama 3 <|start_header_id|>role<|end_header_id|> ... <|eot_id|>, its automatic "Cutting Knowledge Date" system
    message dropped) and re-applied with the decoder's template (as prefill.mixdata.Tok.chat: thinking disabled; Llama date pinned);
  * raw prompts: decoded text, re-encoded with the decoder's tokenizer (+ its BOS if it uses one);
  * responses: decoded text re-encoded; a response that ended its turn gets the decoder's end-of-turn token.
Every source record is checked: re-applying the SOURCE template to the parsed messages must give the source prompt
TEXT back exactly (token ids may differ: the pools concatenate separately tokenised pieces); failures are skipped.
Writes <dst_dir>/<store>/ in the gen_selfdistill.py format (all_tokens/all_offsets/prompt_len/meta.jsonl) plus the
teacher's view for the stage-2 KL (prefill/xteacher.py): the teacher reads its OWN original sequence (native template
and tokens), and every decoder response token is matched to the teacher response token that starts at the same byte
of the response text (both tokenizers are byte-level BPE):
  teacher_tokens.npy / teacher_offsets.npy / teacher_prompt_len.npy   the source sequence as it was generated
  align_tokens.npy / align_offsets.npy (int32)                        per decoder response token: index of the teacher
                                                                      response token starting at the same byte, or -1
The decoder's end-of-turn token is matched to the teacher's. Unmatched positions are left out of the KL.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # noqa: E402

from prefill.mixdata import Tok  # noqa: E402
from prefill.models import eos_ids  # noqa: E402

QWEN_MSG = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n", re.S)
LLAMA_MSG = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.S)


LLAMA_DATE = "26 Jul 2024"  # the Llama 3.1 template's "Today Date" in every data_llama pool / sd store


def chat_ids(tok, msgs):
    """Like prefill.mixdata.Tok.chat, with the Llama template's date pinned to the one the pools were built with."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                   date_string=LLAMA_DATE)
    return tok(text, add_special_tokens=False)["input_ids"]


U2B = {u: b for b, u in bytes_to_unicode().items()}


def tok_bytes(tok, ids):
    """Byte string of every token (byte-level BPE); special tokens -> None."""
    special = set(tok.all_special_ids)
    out = []
    for i, t in zip(ids, tok.convert_ids_to_tokens(ids)):
        out.append(None if i in special or t is None or any(c not in U2B for c in t) else bytes(U2B[c] for c in t))
    return out


def align(tb, db):
    """tb / db: byte strings of the teacher / decoder response tokens (None = special). -> per decoder token the index of
    the teacher token starting at the same byte (-1 if none), up to where the two byte streams agree."""
    starts, pos, tstream = {}, 0, b""
    for k, b in enumerate(tb):
        if b is None:
            break
        starts[pos] = k
        pos += len(b)
        tstream += b
    out, pos, dstream = [], 0, b""
    for b in db:
        if b is None:
            out.append(-1)
            continue
        dstream += b
        ok = pos in starts and tstream[:len(dstream)] == dstream
        out.append(starts[pos] if ok else -1)
        pos += len(b)
    return out


def parse_chat(text):
    pat = QWEN_MSG if "<|im_start|>" in text else LLAMA_MSG
    msgs = [{"role": r, "content": c} for r, c in pat.findall(text)]
    if msgs and msgs[0]["role"] == "system" and msgs[0]["content"].startswith("Cutting Knowledge Date"):
        msgs = msgs[1:]  # Llama 3.1's template adds it to every conversation
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="teacher family's data tree (its sd_* stores)")
    ap.add_argument("--dst_dir", required=True, help="output data tree (decoder tokens)")
    ap.add_argument("--src_tok", required=True)
    ap.add_argument("--dst_tok", required=True)
    ap.add_argument("--stores", default="sd_mix4k,sd_mixlong,sd_mixtest")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (testing)")
    a = ap.parse_args()
    ts, td = AutoTokenizer.from_pretrained(a.src_tok), AutoTokenizer.from_pretrained(a.dst_tok)
    Td = Tok(td)
    src_eos = set(eos_ids(ts))
    dst_eot = td.convert_tokens_to_ids("<|im_end|>" if "<|im_end|>" in td.get_vocab() else "<|eot_id|>")
    for name in a.stores.split(","):
        d = os.path.join(a.src_dir, name)
        toks = np.load(os.path.join(d, "all_tokens.npy"), mmap_mode="r")
        offs = np.load(os.path.join(d, "all_offsets.npy"))
        plen = np.load(os.path.join(d, "prompt_len.npy"))
        meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
        n = len(meta) if not a.limit else min(a.limit, len(meta))
        seqs, plens, metas, bad = [], [], [], 0
        tseqs, tplens, aligns = [], [], []
        for i in range(n):
            ids = np.asarray(toks[offs[i]:offs[i + 1]], dtype=np.int64).tolist()
            P = int(plen[i])
            m = meta[i]
            if m["fmt"] == "chat":
                ptxt = ts.decode(ids[:P], skip_special_tokens=False)
                msgs = parse_chat(ptxt)
                if not msgs or ts.decode(chat_ids(ts, msgs), skip_special_tokens=False) != ptxt:
                    bad += 1
                    continue
                pd = chat_ids(td, msgs)
            else:
                pd = Td.raw(ts.decode(ids[:P], skip_special_tokens=True))
            resp = ids[P:]
            ended = bool(resp) and resp[-1] in src_eos
            rd = Td.enc(ts.decode(resp, skip_special_tokens=True)) + ([dst_eot] if ended else [])
            if not rd:
                bad += 1
                continue
            al = align(tok_bytes(ts, resp), tok_bytes(td, rd))
            if ended:
                al[-1] = len(resp) - 1  # end of turn <-> end of turn
            seqs.append(np.asarray(pd + rd, dtype=np.uint32))  # numpy, not lists: ~26k sequences of up to 8k tokens
            plens.append(len(pd))
            tseqs.append(np.asarray(toks[offs[i]:offs[i + 1]], dtype=np.uint32))
            tplens.append(P)
            aligns.append(np.asarray(al, dtype=np.int32))
            metas.append(dict(m, resp_len=len(rd), src_idx=i, src_prompt_len=P,
                              aligned=round(sum(x >= 0 for x in al) / len(al), 3)))
            if len(seqs) % 2000 == 0:
                print(f"[convert] {name}: {len(seqs)}/{n}", flush=True)
        od = os.path.join(a.dst_dir, name)
        os.makedirs(od, exist_ok=True)
        lens = np.array([len(s) for s in seqs], dtype=np.int64)
        np.save(os.path.join(od, "all_tokens.npy"), np.concatenate([np.asarray(s, dtype=np.uint32) for s in seqs]))
        np.save(os.path.join(od, "all_offsets.npy"), np.concatenate([[0], np.cumsum(lens)]))
        np.save(os.path.join(od, "prompt_len.npy"), np.array(plens, dtype=np.int64))
        tl = np.array([len(x) for x in tseqs], dtype=np.int64)
        np.save(os.path.join(od, "teacher_tokens.npy"), np.concatenate([np.asarray(x, dtype=np.uint32) for x in tseqs]))
        np.save(os.path.join(od, "teacher_offsets.npy"), np.concatenate([[0], np.cumsum(tl)]))
        np.save(os.path.join(od, "teacher_prompt_len.npy"), np.array(tplens, dtype=np.int64))
        al = np.array([len(x) for x in aligns], dtype=np.int64)
        np.save(os.path.join(od, "align_tokens.npy"), np.concatenate([np.asarray(x, dtype=np.int32) for x in aligns]))
        np.save(os.path.join(od, "align_offsets.npy"), np.concatenate([[0], np.cumsum(al)]))
        with open(os.path.join(od, "meta.jsonl"), "w") as f:  # written last: its presence marks a finished store
            for x in metas:
                f.write(json.dumps(x) + "\n")
        r = np.array([x["src_prompt_len"] for x in metas]), np.array(plens)
        print(f"[convert] {name}: {len(seqs)}/{n} records ({bad} skipped), prompt tokens src {r[0].mean():.0f} -> "
              f"dst {r[1].mean():.0f}, ended turns {sum(int(s[-1]) == dst_eot for s in seqs)}, response tokens aligned to the teacher "
              f"{np.mean(np.concatenate(aligns) >= 0):.1%} -> {od}", flush=True)


if __name__ == "__main__":
    main()
