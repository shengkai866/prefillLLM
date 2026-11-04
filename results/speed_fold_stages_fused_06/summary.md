# Speed v2 benchmark (NVIDIA L40S)

## TTFT (ms, median)

| prompt | 8B HF | hybrid HF | 8B lean | hybrid lean | 8B graph | hybrid graph | speedup HF | speedup lean | speedup graph | FLOP ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 406.9 | 198.9 | 410.7 | 108.3 | 411.8 | 105.6 | 2.05x | 3.79x | 3.90x | 6.39x |
| 8192 | 952.4 | 385.4 | 948.4 | 219.1 | 953.2 | 217.5 | 2.47x | 4.33x | 4.38x | 5.99x |
| 16384 | 2139.3 | 873.4 | 2159.6 | 546.9 | 2159.4 | 547.8 | 2.45x | 3.95x | 3.94x | 5.19x |
| 32768 | 5145.1 | - | 5145.9 | 1452.4 | 5101.3 | 1442.7 | - | 3.54x | 3.54x | 4.31x |

Graphed hybrid, per stage (ms): small prefill / folded projector+KV / sink+tail 8B call

- 4096: 49.6 / 21.7 / 35.2
- 8192: 137.3 / 42.7 / 41.0
- 16384: 413.9 / 83.8 / 52.9
- 32768: 1194.8 / 167.9 / 76.3
