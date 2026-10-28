"""
Shared document, several questions: the document prefix (ids [T]) is prefilled ONCE per method, then every question
(its own ids [Q]: "Question: ..." + chat template, ~30-60 tokens) is appended at positions T..T+Q-1 and answered by
greedy decoding of the large model. This is the setting of prefix caching / multi-turn use, where a method whose
cache depends on the question (SpecPrefill: unselected tokens are gone) has to re-prefill or answer from a cache
selected for another question, while a full cache (true or projected) is reused and refined per question.

Configs (MultiQ.doc builds the state, MultiQ.ask answers one question from it):
  big                  large model prefills the document; questions extend its cache
  small                small model alone (its own cache + decoding)
  hyb                  projected cache (sink + projected middle + true last n_tail document tokens); no recompute
  ours:f[:unit]        projected cache + small-model cache; per question the small model scores the middle tokens with
                       the question's queries (cheap: only the question tokens run), the top units are recomputed by
                       the large model TOGETHER with the question tokens, on a copy of the shared cache (recomputed
                       rows do not persist)
  ours-acc:f[:unit]    same, but recomputed rows persist in the shared cache and later questions pick NEW units
                       (progressive refinement; the fraction f is per question)
  spec-agn:f           SpecPrefill without the question: look-ahead from the end of the document, keep f of the chunks,
                       the compact cache answers every question
  spec-q1:f            SpecPrefill scored with the FIRST question (prompt = document + q1); document chunks kept by that
                       score answer every question (the cache is bound to q1)
  spec-each:f          SpecPrefill re-run per question on document + question (nothing is reused; = the single-question
                       benchmark setting)
  ours-each:f[:unit]   our standard single-prompt pipeline re-run per question (nothing reused)
Fractions mean what they mean in scripts/eval_recompute.py (ours: of the middle tokens; spec: of the whole prompt).
"""
import torch

from .fast import kv_to_dynamic_cache
from .kv import greedy_decode, split_prompt
from .recompute import choose_units, extend_at, prefill_kept, segment_ids, spec_importance, spec_select

METHODS = ("big", "small", "hyb", "ours", "ours-acc", "spec-agn", "spec-q1", "spec-each", "ours-each")


def parse_cfg(cfg):
    """-> (method, fraction, unit) with unit None | int block size | 'sent'"""
    s, *rest = cfg.split(":")
    assert s in METHODS, cfg
    f = float(rest[0]) if rest else 0.0
    u = rest[1] if len(rest) > 1 else None
    return s, f, (None if u is None else u if u == "sent" else int(u[1:]))


def with_room(kb, vb, extra):
    """Buffers [L, Hkv, T, D] -> [L, Hkv, T + extra, D] (copy; the new rows are free). Free rows are zeroed: they are
    masked out, but a NaN bit pattern in an uninitialised key would survive the -inf mask."""
    pad = torch.zeros(kb.shape[0], kb.shape[1], extra, kb.shape[3], dtype=kb.dtype, device=kb.device)
    return torch.cat([kb, pad], 2), torch.cat([vb, pad], 2)


class MultiQ:
    def __init__(self, rh, large_hf, small_hf, eos, q_max=128):
        self.rh, self.S, self.L = rh, rh.S, rh.L
        self.large_hf, self.small_hf, self.eos, self.q_max = large_hf, small_hf, eos, q_max

    @torch.no_grad()
    def doc(self, cfg, ids):
        """Prefill the document ids [T] for config cfg -> state (what ask() needs). *-each and spec-q1 do nothing yet."""
        meth, f, u = parse_cfg(cfg)
        T = ids.shape[0]
        st = dict(meth=meth, f=f, u=u, ids=ids, T=T, n_q=0)
        S, L = self.S, self.L
        if meth in ("big", "small"):
            M = L if meth == "big" else S
            kb = torch.zeros(M.n_layers, M.Hkv, T + self.q_max, M.D, dtype=M.embed.dtype, device=ids.device)
            vb = torch.zeros_like(kb)  # free V rows get weight 0, but 0 x NaN = NaN
            M.prefill(ids, kb, vb, logits=False)
            st.update(kb=kb, vb=vb)
        elif meth == "hyb":
            kb, vb, _ = self.rh.prefill(ids)
            st["kb"], st["vb"] = with_room(kb, vb, self.q_max)
        elif meth in ("ours", "ours-acc"):
            # one small-model pass gives both its own cache (for scoring the questions later) and the hidden states
            # for the projector; then the same fold + sink/tail call as FastHybrid.prefill
            s, m, _ = split_prompt(T, self.rh.n_sink, self.rh.n_tail)
            ks = torch.zeros(S.n_layers, S.Hkv, T + self.q_max, S.D, dtype=S.embed.dtype, device=ids.device)
            vs = torch.zeros_like(ks)
            _, hs = S.prefill(ids, ks, vs, collect=True, logits=False)
            kb, vb = self.rh.stage_fold(hs[:, s:s + m], s, T)
            self.rh.stage_tail(ids, kb, vb, s, m)
            st["kb"], st["vb"] = with_room(kb, vb, self.q_max)
            st.update(ks=ks, vs=vs, s=s, m=m, k=min(m, int(round(f * m))),
                      unit=None if u is None else segment_ids(ids[s:s + m], u, self.rh.boundary, self.rh.cap),
                      done=torch.zeros(m, dtype=torch.bool, device=ids.device))
        elif meth == "spec-agn":
            _, imp = spec_importance(S, ids, self.rh.n_lah, self.rh.eos, self.rh.spec_kernel)
            self._keep(st, spec_select(imp, f, self.rh.spec_chunk))
        return st

    def _keep(self, st, pos):
        """Compact large-model cache of the document tokens at positions pos (+ room for a question)."""
        st["pos"] = pos
        st["kb"], st["vb"], _ = prefill_kept(self.L, st["ids"], pos, extra=self.q_max)

    @torch.no_grad()
    def ask_prefill(self, st, q_ids):
        """Append question q_ids [Q] -> (kbuf, vbuf of the cache to decode from, first-token logits, fraction of the
        middle/document tokens the large model computed for this question, RoPE position of the first decoded token)."""
        meth, T = st["meth"], st["T"]
        Q = q_ids.shape[0]
        assert Q <= self.q_max, Q
        dev = q_ids.device
        qpos = torch.arange(T, T + Q, device=dev)
        st["n_q"] += 1
        if meth in ("big", "small", "hyb"):
            M = self.S if meth == "small" else self.L
            lg, _ = extend_at(M, q_ids, qpos, st["kb"], st["vb"])
            return st["kb"][:, :, :T + Q], st["vb"][:, :, :T + Q], lg, 1.0 if meth == "big" else 0.0, None
        if meth in ("ours", "ours-acc"):
            s, m, k = st["s"], st["m"], st["k"]
            _, score = extend_at(self.S, q_ids, qpos, st["ks"], st["vs"], n_q=Q, lo=s, hi=s + m)
            if meth == "ours":
                kb, vb = st["kb"].clone(), st["vb"].clone()
            else:
                kb, vb = st["kb"], st["vb"]
                score = score.masked_fill(st["done"], float("-inf"))
            if k == 0 or (meth == "ours-acc" and bool(st["done"].all())):
                idx = torch.arange(0, device=dev)
            elif st["unit"] is not None:
                idx = choose_units(score, st["unit"], k)
            else:
                idx = torch.topk(score, k).indices.sort().values
            if meth == "ours-acc":
                st["done"][idx] = True
            pos = torch.cat([s + idx, qpos])
            lg, _ = extend_at(self.L, torch.cat([st["ids"][s + idx], q_ids]), pos, kb, vb)
            return kb[:, :, :T + Q], vb[:, :, :T + Q], lg, idx.numel() / max(m, 1), None
        if meth in ("spec-agn", "spec-q1"):
            if meth == "spec-q1" and "kb" not in st:  # first question: select document chunks with its look-ahead
                _, imp = spec_importance(self.S, torch.cat([st["ids"], q_ids]), self.rh.n_lah, self.rh.eos, self.rh.spec_kernel)
                self._keep(st, spec_select(imp, st["f"], self.rh.spec_chunk, T=T))
            K = st["pos"].shape[0]
            rows = torch.arange(K, K + Q, device=dev)
            lg, _ = extend_at(self.L, q_ids, qpos, st["kb"], st["vb"], rows=rows)
            return st["kb"][:, :, :K + Q], st["vb"][:, :, :K + Q], lg, K / T, T + Q
        full = torch.cat([st["ids"], q_ids])
        if meth == "spec-each":
            kb, vb, lg, n = self.rh.prefill_specprefill(full, st["f"])
            return kb, vb, lg, n / full.shape[0], T + Q
        kb, vb, lg, n = self.rh.prefill_sel(full, "attn_small", st["f"], seg=st["u"])  # ours-each
        _, m, _ = split_prompt(full.shape[0], self.rh.n_sink, self.rh.n_tail)
        return kb, vb, lg, n / max(m, 1), None

    @torch.no_grad()
    def ask(self, st, q_ids, max_new):
        """-> (generated token ids, fraction computed by the large model for this question)."""
        kb, vb, lg, frac, pos0 = self.ask_prefill(st, q_ids)
        model = self.small_hf if st["meth"] == "small" else self.large_hf
        out = greedy_decode(model, kv_to_dynamic_cache(kb, vb), lg, max_new, self.eos, pos0=pos0)
        return out, frac
