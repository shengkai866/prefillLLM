# Speed v2 benchmark (NVIDIA L40S)

## Correctness of the fast path

- lean 8B vs HF 8B, first-token KL: 0.00e+00
- held-out continuation KL vs 8B (16 seqs, prompt 2048): reference hybrid 0.1147, fast hybrid 0.1150; KL(reference || fast) 8.40e-04
- top-1 agreement with 8B: reference 0.870, fast 0.871
- CUDA graph vs eager max |logit diff|: 0.00e+00
