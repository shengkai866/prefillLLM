# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 391.6 | 193.9 | 395.6 | 103.0 | 395.3 | 100.2 | 2.02x | 3.84x | 3.95x | 6.39x |
| 8192 | 933.3 | 380.6 | 922.0 | 207.9 | 923.5 | 205.6 | 2.45x | 4.43x | 4.49x | 5.99x |
| 16384 | 2079.0 | 859.1 | 2055.3 | 523.4 | 2067.7 | 522.6 | 2.42x | 3.93x | 3.96x | 5.19x |
| 32768 | 5039.3 | - | 5038.8 | 1383.7 | 5038.5 | 1376.7 | - | 3.64x | 3.66x | 4.31x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 48.4 / 17.6 / 35.2
- 8192: 132.4 / 34.6 / 41.0
- 16384: 400.4 / 67.4 / 52.9
- 32768: 1163.3 / 135.8 / 76.3
