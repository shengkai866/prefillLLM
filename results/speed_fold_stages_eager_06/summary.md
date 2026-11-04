# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 403.7 | 197.6 | 409.9 | 145.4 | 410.0 | 142.4 | 2.04x | 2.82x | 2.88x | 6.44x |
| 8192 | 960.3 | 385.5 | 952.0 | 299.2 | 955.0 | 295.4 | 2.49x | 3.18x | 3.23x | 6.03x |
| 16384 | 2130.4 | 869.8 | 2159.2 | 707.1 | 2159.1 | 706.5 | 2.45x | 3.05x | 3.06x | 5.21x |
| 32768 | 5109.9 | - | 5101.4 | 1765.6 | 5121.8 | 1762.8 | - | 2.89x | 2.91x | 4.33x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 49.3 / 59.7 / 35.3
- 8192: 136.0 / 122.2 / 41.0
- 16384: 412.4 / 246.1 / 52.9
- 32768: 1197.3 / 491.0 / 76.4
