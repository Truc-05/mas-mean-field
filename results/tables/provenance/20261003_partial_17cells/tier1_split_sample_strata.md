# Tier-1 strata recovered by split-sample re-binning (offline, 0 GPU)

Bin on `none` replicates 0–3 (p̂ over N×4 votes, frozen cuts easy ≥ 0.80 / hard ≤ 0.30); evaluate all conditions on the held-out replicates 4–7. Margin M from `none` replicates = `all`. Bins with < 15 questions are reported, not used.

Supersedes the probe-based `easy`/`hard` labels, which do not reproduce (display-shuffle `qi` convention differs between `p2_probe.py:56` and `exp6c_boundary.py:239`). Within-cell contrasts were never affected.

## Recovered strata

**ΔD pooled is PRIMARY** (ratified 2026-09-30): it sums the pair-correlation numerator and denominator over ALL questions, so a saturated question contributes num = den = 0 and drops out arithmetically. `ΔD mean-of-ratios` — the frozen per-question estimator — is kept as the legacy comparison, but in the easy strata it is computable on as few as 4 of 45 questions (`n_D`), and that surviving subset is selected on error variance, the very quantity D measures. Both are shown so the choice stays visible. The frozen 4×2 substrate keeps mean-of-ratios for continuity.

| model | benchmark | regime | n_q | A_none | A_shared | A_indep | shared−none | indep−none | dA (C1) | **ΔD pooled** | 95% CI | sig | ΔD mean-of-ratios (legacy) | 95% CI | sig | n_D | \|ΔA\|~M (C2) | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama | aqua | easy | 4 | *insufficient (< 15)* | | | | | | | | | | | | | | |
| llama | aqua | hard | 42 | 0.137 | 0.446 | 0.524 | +0.310 | +0.387 | -0.077 | **+0.552** | [+0.417,+0.672] | ✓ | +0.411 | [+0.271,+0.557] | ✓ | 35 | -0.272 | [-0.508,-0.009] |
| llama | arc | easy | 34 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | **+0.733** | [+0.365,+1.107] | – | +0.849 | [+0.591,+1.107] | ✓ | 2 | +0.000 | [+0.000,+0.000] |
| llama | arc | hard | 25 | 0.060 | 0.360 | 0.380 | +0.300 | +0.320 | -0.020 | **+0.283** | [+0.166,+0.393] | – | +0.245 | [+0.057,+0.419] | ✓ | 21 | -0.065 | [-0.269,+0.141] |
| llama | truthfulqa | easy | 41 | 1.000 | 0.994 | 1.000 | -0.006 | +0.000 | -0.006 | **+0.489** | [-0.093,+1.035] | – | +0.152 | [-0.148,+0.582] | – | 4 | -0.193 | [-0.504,+0.000] |
| llama | truthfulqa | hard | 20 | 0.000 | 0.438 | 0.400 | +0.438 | +0.400 | +0.037 | **+0.205** | [+0.077,+0.361] | – | +0.280 | [+0.132,+0.457] | ✓ | 16 | +0.128 | [-0.059,+0.319] |
| qwen | aqua | easy | 29 | 1.000 | 0.966 | 0.931 | -0.034 | -0.069 | +0.034 | **+0.092** | [-0.037,+0.237] | – | +0.080 | [-0.084,+0.281] | – | 7 | -0.461 | [-0.974,+0.000] |
| qwen | aqua | hard | 39 | 0.032 | 0.256 | 0.295 | +0.224 | +0.263 | -0.038 | **+0.538** | [+0.364,+0.688] | ✓ | +0.406 | [+0.251,+0.557] | ✓ | 27 | +0.136 | [-0.070,+0.379] |
| qwen | arc | easy | 41 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | **+0.251** | [+0.107,+1.067] | – | +0.638 | [+0.210,+1.067] | ✓ | 2 | +0.000 | [+0.000,+0.000] |
| qwen | arc | hard | 33 | 0.008 | 0.152 | 0.136 | +0.144 | +0.129 | +0.015 | **+0.686** | [+0.439,+0.893] | – | +0.505 | [+0.201,+0.800] | ✓ | 15 | -0.059 | [-0.241,+0.162] |
| qwen | bbh | easy | 32 | 1.000 | 0.789 | 0.812 | -0.211 | -0.188 | -0.023 | **+0.360** | [+0.120,+0.637] | – | +0.373 | [+0.125,+0.644] | ✓ | 15 | -0.305 | [-1.525,+0.334] |
| qwen | bbh | hard | 33 | 0.000 | 0.068 | 0.061 | +0.068 | +0.061 | +0.008 | **+0.486** | [+0.201,+0.753] | – | +0.474 | [+0.231,+0.737] | ✓ | 12 | -0.178 | [-0.566,+0.136] |
| qwen | commonsenseqa | easy | 41 | 1.000 | 0.994 | 0.994 | -0.006 | -0.006 | +0.000 | **+0.148** | [+0.040,+0.952] | – | +0.428 | [-0.014,+0.906] | – | 6 | +0.000 | [+0.000,+0.000] |
| qwen | commonsenseqa | hard | 26 | 0.038 | 0.192 | 0.221 | +0.154 | +0.183 | -0.029 | **+0.442** | [+0.207,+0.655] | – | +0.349 | [+0.190,+0.511] | ✓ | 14 | -0.186 | [-0.450,+0.030] |
| qwen | mmlu | easy | 32 | 1.000 | 0.891 | 0.961 | -0.109 | -0.039 | -0.070 | **+0.255** | [+0.122,+0.410] | – | +0.301 | [+0.129,+0.514] | ✓ | 14 | -0.831 | [-1.406,-0.266] |
| qwen | mmlu | hard | 33 | 0.030 | 0.197 | 0.129 | +0.167 | +0.098 | +0.068 | **+0.457** | [+0.248,+0.666] | – | +0.361 | [+0.167,+0.561] | ✓ | 19 | -0.215 | [-0.514,+0.031] |
| qwen | strategyqa | easy | 43 | 1.000 | 0.948 | 0.977 | -0.052 | -0.023 | -0.029 | **+0.512** | [+0.255,+0.702] | – | +0.414 | [+0.184,+0.672] | ✓ | 6 | +0.104 | [-0.110,+0.434] |
| qwen | strategyqa | hard | 34 | 0.007 | 0.515 | 0.529 | +0.507 | +0.522 | -0.015 | **+0.766** | [+0.630,+0.888] | ✓ | +0.730 | [+0.584,+0.875] | ✓ | 19 | -0.012 | [-0.432,+0.367] |
| qwen | truthfulqa | easy | 44 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | **-0.133** | — | – | — | — | – | 0 | +0.000 | [+0.000,+0.000] |
| qwen | truthfulqa | hard | 30 | 0.008 | 0.308 | 0.325 | +0.300 | +0.317 | -0.017 | **+0.361** | [+0.184,+0.555] | – | +0.446 | [+0.257,+0.643] | ✓ | 13 | -0.059 | [-0.192,+0.067] |

## Bin migration — old probe label → recovered label

| model | benchmark | old easy → easy/mid/hard | old hard → easy/mid/hard |
|---|---|---|---|
| llama | aqua | 4/21/15 | 0/13/27 |
| llama | arc | 31/6/3 | 3/15/22 |
| llama | truthfulqa | 29/8/3 | 12/11/17 |
| qwen | aqua | 26/5/9 | 3/7/30 |
| qwen | arc | 38/2/0 | 3/4/33 |
| qwen | bbh | 20/7/13 | 12/8/20 |
| qwen | commonsenseqa | 33/6/1 | 8/7/25 |
| qwen | mmlu | 26/6/8 | 6/9/25 |
| qwen | strategyqa | 36/1/3 | 7/2/31 |
| qwen | truthfulqa | 38/2/0 | 6/4/30 |

## Pooled per model (question-weighted mean over recovered strata)

| model | cells | n_q | mean A_none | mean dA | **ΔD pooled** | ΔD mean-of-ratios (legacy) | mean C2 slope |
|---|---|---|---|---|---|---|---|
| llama | 5 | 162 | 0.508 | -0.020 | **+0.490** | +0.396 | -0.113 |
| qwen | 14 | 490 | 0.543 | -0.007 | **+0.363** | +0.432 | -0.126 |
