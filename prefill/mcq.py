"""Multiple-choice scoring from first-token logits (QuALITY, LongHealth, CLIPPER, ContractNLI): the answer is the
letter whose first token has the highest logit; every letter has a few surface forms ("A", " A", "(A") and the best of
them counts. `n` letters (2..5) are supported: A/B (true-false), A/B/C (NLI), A-D (QuALITY), A-E (LongHealth)."""
import torch

LETTERS = "ABCDE"


def letter_token_ids(tok, n=4):
    """[n][k] first-token ids of the surface forms of each of the first n letters (in the model's own vocabulary)."""
    out = []
    for L in LETTERS[:n]:
        ids = set()
        for s in (L, " " + L, "(" + L, L + ")", L + "."):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                ids.add(t[0])
        out.append(sorted(ids))
    return out


def choose_letter(logits, letter_ids):
    """logits [1, V] or [V] -> index 0..n-1 of the predicted letter."""
    lg = logits.reshape(-1).float()
    best = torch.stack([lg[ids].max() for ids in letter_ids])
    return int(best.argmax())
