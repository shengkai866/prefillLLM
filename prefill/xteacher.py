"""
Teacher of a cross-family REVERSE pair (e.g. Llama-3.1-8B prefills -> Qwen3-1.7B decodes; scripts/convert_sd_cross.py).

The stage-2 sequences are in the decoder's tokens, but the teacher (the prefilling 8B, "small" in code) reads its OWN
original sequence: its native chat template and tokens, i.e. exactly what it saw when it wrote the answer. Its logits
at the response tokens matched by byte position (align_*: decoder response token j <-> teacher response token k that
starts at the same byte) are moved onto the decoder's vocabulary over the tokens both vocabularies share
(small.teacher_vocab, prefill/tokalign.shared_vocab): shared tokens get the teacher's logit, every other token -inf.
kv.kl_teacher_student renormalises both distributions over the shared tokens; unmatched positions are masked out.
"""
import os

import numpy as np
import torch

from . import config as C


class NativeTeacher:
    def __init__(self, name):
        d = os.path.join(C.DATA_DIR, name)
        self.tok = np.load(os.path.join(d, "teacher_tokens.npy"), mmap_mode="r")
        self.off = np.load(os.path.join(d, "teacher_offsets.npy"))
        self.plen = np.load(os.path.join(d, "teacher_prompt_len.npy"))
        self.al = np.load(os.path.join(d, "align_tokens.npy"), mmap_mode="r")
        self.aoff = np.load(os.path.join(d, "align_offsets.npy"))

    @staticmethod
    def load(name):
        """NativeTeacher for this store if it has a teacher view (cross-family reverse pair), else None."""
        return NativeTeacher(name) if os.path.exists(os.path.join(C.DATA_DIR, name, "teacher_tokens.npy")) else None

    @torch.no_grad()
    def logits(self, model, i, device):
        """-> (t_logits [1, m, V_decoder], mask [1, m] bool) for the m response tokens of decoder sequence i.
        Unmatched rows repeat a matched row (finite pattern intact) and are False in mask."""
        ids = torch.from_numpy(np.array(self.tok[self.off[i]:self.off[i + 1]], dtype=np.int64))[None].to(device)
        P = int(self.plen[i])
        al = torch.from_numpy(np.array(self.al[self.aoff[i]:self.aoff[i + 1]], dtype=np.int64)).to(device)
        out = model(input_ids=ids, use_cache=False, logits_to_keep=ids.shape[1] - P + 1)
        lg = out.logits[0, :-1]  # [m_teacher, V_teacher]: row k predicts teacher response token k
        mask = al >= 0
        rows = lg[al.clamp(min=0)]  # [m, V_teacher]
        if bool(mask.any()):
            rows = torch.where(mask[:, None], rows, rows[mask.nonzero()[0, 0]][None])
        li, si, V = model.teacher_vocab
        full = rows.new_full((rows.shape[0], V), float("-inf"))
        full[:, li.to(device)] = rows[:, si.to(device)]
        return full[None], mask[None]


def masked_mean(x, mask):
    """Mean of x [B, m] over mask [B, m] (None = all)."""
    if mask is None:
        return x.mean()
    mask = mask.to(x.device)
    return (x * mask).sum() / mask.sum().clamp(min=1)
