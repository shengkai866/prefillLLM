# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 408.8 | 262.1 | 410.8 | 203.9 | 410.1 | 200.5 | 1.56x | 2.01x | 2.04x | 3.45x |
| 8192 | 938.7 | 523.6 | 940.5 | 428.3 | 936.8 | 425.8 | 1.79x | 2.20x | 2.20x | 3.43x |
| 16384 | 2139.1 | - | 2159.3 | 987.9 | 2159.2 | 981.0 | - | 2.19x | 2.20x | 3.32x |
| 32768 | 5080.9 | - | 5082.8 | 2284.1 | 5152.1 | 2287.9 | - | 2.23x | 2.25x | 3.14x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 96.4 / 73.2 / 35.2
- 8192: 241.3 / 147.2 / 41.0
- 16384: 643.8 / 296.2 / 53.0
- 32768: 1623.8 / 595.0 / 76.4
