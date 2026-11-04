# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 401.5 | 263.9 | 409.1 | 165.8 | 406.2 | 163.9 | 1.52x | 2.47x | 2.48x | 3.44x |
| 8192 | 948.7 | 524.8 | 936.1 | 352.7 | 945.2 | 346.8 | 1.81x | 2.65x | 2.73x | 3.42x |
| 16384 | 2134.2 | - | 2159.4 | 828.1 | 2159.3 | 826.5 | - | 2.61x | 2.61x | 3.31x |
| 32768 | 5084.5 | - | 5073.3 | 1979.9 | 5135.3 | 1980.2 | - | 2.56x | 2.59x | 3.13x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 97.6 / 34.3 / 35.3
- 8192: 241.9 / 67.9 / 41.0
- 16384: 645.4 / 136.3 / 52.9
- 32768: 1630.0 / 274.4 / 76.3
