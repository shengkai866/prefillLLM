# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 393.7 | 261.7 | 397.5 | 155.9 | 398.7 | 153.3 | 1.50x | 2.55x | 2.60x | 3.44x |
| 8192 | 926.1 | 519.6 | 920.7 | 322.0 | 923.3 | 319.9 | 1.78x | 2.86x | 2.89x | 3.42x |
| 16384 | 2069.8 | - | 2068.7 | 775.0 | 2099.0 | 769.9 | - | 2.67x | 2.73x | 3.31x |
| 32768 | 5058.2 | - | 5038.7 | 1858.0 | 5038.3 | 1840.0 | - | 2.71x | 2.74x | 3.13x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 95.4 / 20.5 / 35.2
- 8192: 238.3 / 39.7 / 41.0
- 16384: 636.9 / 87.3 / 52.9
- 32768: 1595.2 / 173.9 / 76.3
