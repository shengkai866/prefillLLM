# Speed v2 benchmark (NVIDIA H100 80GB HBM3)

## Correctness of the fast path

- lean 8B vs HF 8B, first-token KL: 0.00e+00
- held-out continuation KL vs 8B (16 seqs, prompt 2048): reference hybrid 0.1165, fast hybrid 0.1169; KL(reference || fast) 7.80e-04
- top-1 agreement with 8B: reference 0.860, fast 0.862
- CUDA graph vs eager max |logit diff|: 0.00e+00

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 26.1 | 77.1 | 24.3 | 33.3 | 21.5 | 20.9 | 0.34x | 0.73x | 1.03x | 4.07x |
| 1024 | 40.6 | 75.3 | 40.0 | 33.7 | 37.6 | 27.4 | 0.54x | 1.19x | 1.37x | 5.37x |
| 2048 | 78.6 | 78.2 | 78.6 | 44.4 | 76.5 | 39.9 | 1.01x | 1.77x | 1.92x | 6.23x |
| 4096 | 157.3 | 115.0 | 158.5 | 74.3 | 157.7 | 69.7 | 1.37x | 2.13x | 2.26x | 6.44x |
| 8192 | 342.4 | 197.8 | 341.6 | 141.4 | 344.4 | 136.7 | 1.73x | 2.42x | 2.52x | 6.03x |
| 16384 | 798.6 | 397.9 | 803.1 | 307.4 | 802.3 | 303.0 | 2.01x | 2.61x | 2.65x | 5.21x |
| 32768 | 2078.8 | 921.5 | 2082.5 | 778.5 | 2075.9 | 767.3 | 2.26x | 2.68x | 2.71x | 4.33x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 512: 6.0 / 2.1 / 12.8
- 1024: 9.4 / 4.4 / 13.6
- 2048: 15.9 / 8.8 / 15.1
- 4096: 32.6 / 18.7 / 18.6
- 8192: 74.2 / 37.7 / 25.2
- 16384: 191.2 / 75.5 / 38.4
- 32768: 561.5 / 151.3 / 63.0

## Needle-in-a-haystack: which part of the projected KV loses the needle

| variant | 2048 | 3584 |
|---|---|---|
| hybrid | 0.50 | 0.57 |
| all_true | 1.00 | 1.00 |
| needle_true | 1.00 | 1.00 |
| needle_trueK | 0.90 | 0.97 |
| needle_trueV | 0.67 | 0.77 |
| needle_true_early | 0.57 | 0.53 |
| needle_true_mid | 1.00 | 1.00 |
| needle_true_late | 1.00 | 0.97 |
| layers_true_early | 0.57 | 0.57 |
| layers_true_mid | 1.00 | 1.00 |
| layers_true_late | 1.00 | 0.97 |

early / mid / late = layers L0-11, L12-23, L24-35.

K/V cosine to the true 8B cache, mean over layer groups (needle tokens vs other middle tokens):

| len | part | L0-11 | L12-23 | L24-35 |
|---|---|---|---|---|
| 2048 | needle_K | 0.894 | 0.797 | 0.749 |
| 2048 | other_K | 0.912 | 0.827 | 0.803 |
| 2048 | needle_V | 0.496 | 0.508 | 0.389 |
| 2048 | other_V | 0.557 | 0.537 | 0.459 |
| 3584 | needle_K | 0.898 | 0.811 | 0.761 |
| 3584 | other_K | 0.915 | 0.834 | 0.811 |
| 3584 | needle_V | 0.503 | 0.521 | 0.393 |
| 3584 | other_V | 0.565 | 0.547 | 0.466 |

Example hybrid failures:

- len 2048, code 002496, depth 0.48: `'002466'`
- len 2048, code 335178, depth 0.36: `'325178'`
- len 2048, code 292526, depth 0.13: `'292266'`
- len 2048, code 693139, depth 0.81: `'693397'`
- len 2048, code 051519, depth 0.37: `'051119'`
- len 2048, code 219043, depth 0.8: `'211943'`
- len 2048, code 351382, depth 0.14: `'3511823'`
- len 2048, code 826219, depth 0.91: `'8266191919'`

## Per-layer fidelity on FineWeb-Edu (cosine to truth, positions >= 8)

| projector | quantity | L0-0 | L1-11 | L12-23 | L24-35 |
|---|---|---|---|---|---|
| stage2 | h | 0.213 | 0.760 | 0.761 | 0.743 |
| stage2 | k | 0.995 | 0.910 | 0.841 | 0.822 |
| stage2 | v | 0.324 | 0.591 | 0.545 | 0.473 |
| stage1 | h | 0.522 | 0.819 | 0.803 | 0.791 |
| stage1 | k | 0.997 | 0.943 | 0.879 | 0.871 |
| stage1 | v | 0.719 | 0.675 | 0.613 | 0.542 |
