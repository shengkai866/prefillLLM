# prefillLLM evaluation (NVIDIA H100 80GB HBM3)

## Speed: time-to-first-token (median, ms) vs prompt length

| prompt_len | big | hybrid_stage1_s8_t64 | hybrid_stage2_s8_t64 | small |
|---|---|---|---|---|
| 512 | 25.7 | 83.7 | 83.5 | 18.3 |
| 1024 | 40.2 | 85.3 | 83.7 | 18.2 |
| 2048 | 77.5 | 89.6 | 89.8 | 18.6 |
| 4096 | 155.5 | 121.0 | 121.1 | 35.4 |

Decode latency per token (ms): big=21.8, small=15.6, hybrid_stage1_s8_t64=21.7, hybrid_stage2_s8_t64=22.0

Hybrid TTFT breakdown (ms) at the longest prompt:

- hybrid_stage1_s8_t64: sink=24.0, small=37.4, projector=9.6, kv=20.7, tail=29.3
- hybrid_stage2_s8_t64: sink=24.2, small=37.4, projector=9.6, kv=20.7, tail=29.2

## Held-out continuation quality (256 tokens after the prompt)

| source | prompt_len | config | NLL | PPL | KL vs big | top-1 agree |
|---|---|---|---|---|---|---|
| fineweb_edu | 512 | big | 2.3526 | 10.513 | 0.0000 | 1.000 |
| fineweb_edu | 512 | small | 2.9814 | 19.716 | 0.7808 | 0.634 |
| fineweb_edu | 512 | hybrid_stage1_s8_t64 | 2.4861 | 12.015 | 0.1745 | 0.842 |
| fineweb_edu | 512 | hybrid_stage2_s8_t64 | 2.3915 | 10.930 | 0.0729 | 0.888 |
| fineweb_edu | 512 | hybrid_stage2_s16_t128 | 2.3716 | 10.715 | 0.0509 | 0.906 |
| fineweb_edu | 512 | hybrid_stage2_s4_t32 | 2.4037 | 11.064 | 0.0917 | 0.877 |
| fineweb_edu | 1024 | big | 2.2981 | 9.955 | 0.0000 | 1.000 |
| fineweb_edu | 1024 | small | 2.9603 | 19.304 | 0.8113 | 0.634 |
| fineweb_edu | 1024 | hybrid_stage1_s8_t64 | 2.4566 | 11.665 | 0.1893 | 0.838 |
| fineweb_edu | 1024 | hybrid_stage2_s8_t64 | 2.3508 | 10.494 | 0.0839 | 0.884 |
| fineweb_edu | 1024 | hybrid_stage2_s16_t128 | 2.3390 | 10.371 | 0.0670 | 0.896 |
| fineweb_edu | 1024 | hybrid_stage2_s4_t32 | 2.3656 | 10.651 | 0.1011 | 0.871 |
| fineweb_edu | 2048 | big | 2.2865 | 9.840 | 0.0000 | 1.000 |
| fineweb_edu | 2048 | small | 2.9027 | 18.222 | 0.7752 | 0.641 |
| fineweb_edu | 2048 | hybrid_stage1_s8_t64 | 2.4943 | 12.114 | 0.2383 | 0.824 |
| fineweb_edu | 2048 | hybrid_stage2_s8_t64 | 2.3508 | 10.494 | 0.1068 | 0.867 |
| fineweb_edu | 2048 | hybrid_stage2_s16_t128 | 2.3359 | 10.338 | 0.0871 | 0.880 |
| fineweb_edu | 2048 | hybrid_stage2_s4_t32 | 2.3674 | 10.670 | 0.1242 | 0.857 |
| fineweb_edu | 3584 | big | 2.1748 | 8.800 | 0.0000 | 1.000 |
| fineweb_edu | 3584 | small | 2.8901 | 17.995 | 0.8435 | 0.639 |
| fineweb_edu | 3584 | hybrid_stage1_s8_t64 | 2.4228 | 11.277 | 0.2940 | 0.820 |
| fineweb_edu | 3584 | hybrid_stage2_s8_t64 | 2.2670 | 9.650 | 0.1425 | 0.864 |
| fineweb_edu | 3584 | hybrid_stage2_s16_t128 | 2.2520 | 9.507 | 0.1255 | 0.874 |
| fineweb_edu | 3584 | hybrid_stage2_s4_t32 | 2.2830 | 9.806 | 0.1578 | 0.859 |
| slimpajama | 512 | big | 2.3047 | 10.022 | 0.0000 | 1.000 |
| slimpajama | 512 | small | 3.0021 | 20.127 | 0.8751 | 0.619 |
| slimpajama | 512 | hybrid_stage1_s8_t64 | 2.4934 | 12.103 | 0.2374 | 0.827 |
| slimpajama | 512 | hybrid_stage2_s8_t64 | 2.3719 | 10.718 | 0.1067 | 0.871 |
| slimpajama | 512 | hybrid_stage2_s16_t128 | 2.3473 | 10.457 | 0.0760 | 0.892 |
| slimpajama | 512 | hybrid_stage2_s4_t32 | 2.3943 | 10.960 | 0.1256 | 0.859 |
| slimpajama | 1024 | big | 2.3679 | 10.675 | 0.0000 | 1.000 |
| slimpajama | 1024 | small | 3.0230 | 20.552 | 0.8256 | 0.631 |
| slimpajama | 1024 | hybrid_stage1_s8_t64 | 2.6135 | 13.647 | 0.3005 | 0.800 |
| slimpajama | 1024 | hybrid_stage2_s8_t64 | 2.4644 | 11.756 | 0.1204 | 0.864 |
| slimpajama | 1024 | hybrid_stage2_s16_t128 | 2.4450 | 11.531 | 0.1003 | 0.877 |
| slimpajama | 1024 | hybrid_stage2_s4_t32 | 2.4782 | 11.920 | 0.1371 | 0.852 |
| slimpajama | 2048 | big | 2.2618 | 9.601 | 0.0000 | 1.000 |
| slimpajama | 2048 | small | 2.8775 | 17.770 | 0.7969 | 0.648 |
| slimpajama | 2048 | hybrid_stage1_s8_t64 | 2.5579 | 12.909 | 0.3518 | 0.797 |
| slimpajama | 2048 | hybrid_stage2_s8_t64 | 2.3610 | 10.602 | 0.1422 | 0.860 |
| slimpajama | 2048 | hybrid_stage2_s16_t128 | 2.3416 | 10.398 | 0.1178 | 0.874 |
| slimpajama | 2048 | hybrid_stage2_s4_t32 | 2.3810 | 10.816 | 0.1647 | 0.851 |
| slimpajama | 3584 | big | 2.1525 | 8.606 | 0.0000 | 1.000 |
| slimpajama | 3584 | small | 2.7326 | 15.372 | 0.7237 | 0.668 |
| slimpajama | 3584 | hybrid_stage1_s8_t64 | 2.5284 | 12.534 | 0.4391 | 0.773 |
| slimpajama | 3584 | hybrid_stage2_s8_t64 | 2.2904 | 9.879 | 0.1994 | 0.837 |
| slimpajama | 3584 | hybrid_stage2_s16_t128 | 2.2685 | 9.665 | 0.1741 | 0.849 |
| slimpajama | 3584 | hybrid_stage2_s4_t32 | 2.3119 | 10.094 | 0.2177 | 0.827 |

## GSM8K (few-shot, greedy, non-thinking mode)

| config | n | accuracy | prompt tokens | TTFT (s) | total (s) | new tokens |
|---|---|---|---|---|---|---|
| big | 150 | 0.940 | 1353 | 0.055 | 3.18 | 142 |
| small | 150 | 0.100 | 1353 | 0.018 | 1.20 | 75 |
| hybrid_stage1_s8_t64 | 150 | 0.700 | 1353 | 0.085 | 5.03 | 220 |
| hybrid_stage2_s8_t64 | 150 | 0.900 | 1353 | 0.085 | 4.20 | 182 |

## Needle in a haystack (exact passcode retrieval)

| config | target_len | prompt tokens | accuracy | TTFT (s) |
|---|---|---|---|---|
| big | 1024 | 992 | 1.000 | 0.040 |
| big | 2048 | 2016 | 1.000 | 0.078 |
| big | 3584 | 3552 | 1.000 | 0.137 |
| small | 1024 | 992 | 1.000 | 0.018 |
| small | 2048 | 2016 | 1.000 | 0.019 |
| small | 3584 | 3552 | 1.000 | 0.031 |
| hybrid_stage1_s8_t64 | 1024 | 992 | 0.000 | 0.084 |
| hybrid_stage1_s8_t64 | 2048 | 2016 | 0.000 | 0.089 |
| hybrid_stage1_s8_t64 | 3584 | 3552 | 0.000 | 0.113 |
| hybrid_stage2_s8_t64 | 1024 | 992 | 0.467 | 0.084 |
| hybrid_stage2_s8_t64 | 2048 | 2016 | 0.633 | 0.089 |
| hybrid_stage2_s8_t64 | 3584 | 3552 | 0.700 | 0.112 |
| hybrid_stage2_s16_t128 | 1024 | 992 | 0.500 | 0.085 |
| hybrid_stage2_s16_t128 | 2048 | 2016 | 0.600 | 0.089 |
| hybrid_stage2_s16_t128 | 3584 | 3552 | 0.667 | 0.118 |
| hybrid_stage2_s4_t32 | 1024 | 992 | 0.500 | 0.085 |
| hybrid_stage2_s4_t32 | 2048 | 2016 | 0.600 | 0.089 |
| hybrid_stage2_s4_t32 | 3584 | 3552 | 0.733 | 0.112 |

## Oracle: single-layer KV replacement (stage1, prompt 2048, KL vs true, no tail recompute)

none projected = 0.0007; all layers projected = 0.4582

| layer | KL |
|---|---|
| 0 | 0.0010 |
| 1 | 0.0010 |
| 2 | 0.0015 |
| 3 | 0.0018 |
| 4 | 0.0016 |
| 5 | 0.0014 |
| 6 | 0.0014 |
| 7 | 0.0015 |
| 8 | 0.0015 |
| 9 | 0.0044 |
| 10 | 0.0017 |
| 11 | 0.0016 |
| 12 | 0.0023 |
| 13 | 0.0036 |
| 14 | 0.0023 |
| 15 | 0.0045 |
| 16 | 0.0031 |
| 17 | 0.0081 |
| 18 | 0.0054 |
| 19 | 0.0048 |
| 20 | 0.0062 |
| 21 | 0.0045 |
| 22 | 0.0076 |
| 23 | 0.0055 |
| 24 | 0.0068 |
| 25 | 0.0020 |
| 26 | 0.0035 |
| 27 | 0.0031 |
| 28 | 0.0018 |
| 29 | 0.0039 |
| 30 | 0.0023 |
| 31 | 0.0029 |
| 32 | 0.0034 |
| 33 | 0.0074 |
| 34 | 0.0075 |
| 35 | 0.0019 |

## Oracle: single-layer KV replacement (stage2, prompt 2048, KL vs true, no tail recompute)

none projected = 0.0007; all layers projected = 0.2020

| layer | KL |
|---|---|
| 0 | 0.0018 |
| 1 | 0.0015 |
| 2 | 0.0022 |
| 3 | 0.0030 |
| 4 | 0.0024 |
| 5 | 0.0028 |
| 6 | 0.0021 |
| 7 | 0.0018 |
| 8 | 0.0020 |
| 9 | 0.0042 |
| 10 | 0.0018 |
| 11 | 0.0020 |
| 12 | 0.0031 |
| 13 | 0.0039 |
| 14 | 0.0027 |
| 15 | 0.0052 |
| 16 | 0.0039 |
| 17 | 0.0077 |
| 18 | 0.0047 |
| 19 | 0.0058 |
| 20 | 0.0061 |
| 21 | 0.0056 |
| 22 | 0.0078 |
| 23 | 0.0048 |
| 24 | 0.0060 |
| 25 | 0.0027 |
| 26 | 0.0037 |
| 27 | 0.0032 |
| 28 | 0.0027 |
| 29 | 0.0042 |
| 30 | 0.0019 |
| 31 | 0.0033 |
| 32 | 0.0024 |
| 33 | 0.0052 |
| 34 | 0.0057 |
| 35 | 0.0024 |
