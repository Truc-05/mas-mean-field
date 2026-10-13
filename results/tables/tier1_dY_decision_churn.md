# Tier-1 decision churn (ΔY) — the ΔY form of the C1 contrast at breadth

Cells: **66** regime-strata, pooled over the English-rebuild 72-cell grid (`bbh` excluded). Substrate: recovered split-sample strata (`tier1_split_sample_strata.discover`/`load_cell`/`phat_split`, frozen .80/.30 cuts). Evaluated on the held-out replicates only (bin 4 / eval the rest). Estimators verbatim from `analysis_B_matched_mechanism.py:51-61`; `plurality_prediction` imported from the frozen module.

**SCOPE.** Tier-1 has no copying arm, so this does NOT replicate C1's matched-D mechanism-non-equivalence (that stays with `analysis_B`/`analysis_C` on qwen). It establishes the weaker claim: raising dependence churns decisions while leaving net accuracy flat.

**C_within / dC.** `C_within = (C_shared_within + C_indep_within)/2` is the all-pairs decision-churn rate WITHIN a single arm across the held-out replicates — the churn rate sampling noise alone produces with no shared/indep swap at all. `dC = ΔY − C_within` is the churn attributable to the swap net of that baseline; this is the primary audit for whether the `ΔY > 0` headline reflects the treatment or just resampling variance.

`strict` = tie→-1 (analysis_B convention, **primary**). `tiebreak` = seeded random tie-break (boundary-runner convention).

## Pooled over cells — `strict`

| regime | cells | mean dA | mean ΔY | ΔY sig>0 | mean C_within | mean dC | dC sig>0 | mean H_repair |
|---|---|---|---|---|---|---|---|---|
| easy | 31 | -0.009 | 0.019 | 2/31 | 0.014 | +0.004 | 1/31 | +0.009 |
| hard | 35 | -0.003 | 0.270 | 35/35 | 0.256 | +0.014 | 7/35 | +0.003 |

## Pooled over cells — `tiebreak`

| regime | cells | mean dA | mean ΔY | ΔY sig>0 | mean C_within | mean dC | dC sig>0 | mean H_repair |
|---|---|---|---|---|---|---|---|---|
| easy | 31 | -0.010 | 0.018 | 2/31 | 0.014 | +0.005 | 0/31 | +0.010 |
| hard | 35 | -0.010 | 0.245 | 35/35 | 0.233 | +0.012 | 6/35 | +0.010 |

## Per cell

| model | benchmark | regime | n_q | dA | ΔY | C_within | dC | dC 95% CI | H_repair | ΔY tiebreak | dC tiebreak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | aqua | easy | 23 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| deepseek | aqua | hard | 31 | -0.040 | 0.395* | 0.427 | -0.032 | [-0.121,+0.054] | +0.040 | 0.403 | -0.043 |
| deepseek | arc | easy | 27 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| deepseek | arc | hard | 29 | +0.000 | 0.198* | 0.184 | +0.014 | [-0.043,+0.075] | +0.000 | 0.147 | -0.006 |
| deepseek | commonsenseqa | easy | 27 | -0.009 | 0.009 | 0.009 | +0.000 | [+0.000,+0.000] | +0.009 | 0.009 | +0.000 |
| deepseek | commonsenseqa | hard | 31 | -0.040 | 0.323* | 0.304 | +0.019 | [-0.016,+0.051] | +0.040 | 0.298 | +0.022 |
| deepseek | mmlu | easy | 18 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| deepseek | mmlu | hard | 48 | +0.057 | 0.312* | 0.332 | -0.019 | [-0.073,+0.033] | -0.057 | 0.297 | -0.017 |
| deepseek | strategyqa | easy | 18 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| deepseek | strategyqa | hard | 16 | +0.000 | 0.250* | 0.266 | -0.016 | [-0.089,+0.047] | +0.000 | 0.250 | +0.010 |
| deepseek | truthfulqa | easy | 23 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| deepseek | truthfulqa | hard | 28 | -0.036 | 0.384* | 0.336 | +0.048 | [-0.033,+0.134] | +0.036 | 0.321 | +0.051 |
| gemma | aqua | easy | 15 | +0.033 | 0.067 | 0.056 | +0.011 | [+0.000,+0.033] | -0.033 | 0.067 | +0.011 |
| gemma | aqua | hard | 51 | +0.020 | 0.289* | 0.243 | +0.046* | [+0.010,+0.085] | -0.020 | 0.265 | +0.046 |
| gemma | arc | easy | 44 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| gemma | arc | hard | 29 | +0.009 | 0.216* | 0.181 | +0.034 | [-0.006,+0.086] | -0.009 | 0.216 | +0.037 |
| gemma | commonsenseqa | easy | 40 | -0.019 | 0.019 | 0.017 | +0.002 | [+0.000,+0.006] | +0.019 | 0.019 | +0.002 |
| gemma | commonsenseqa | hard | 30 | +0.000 | 0.217* | 0.172 | +0.044 | [-0.003,+0.094] | +0.000 | 0.217 | +0.056 |
| gemma | mmlu | easy | 23 | -0.109* | 0.130* | 0.083 | +0.047 | [+0.000,+0.120] | +0.109* | 0.130 | +0.047 |
| gemma | mmlu | hard | 51 | +0.034 | 0.284* | 0.227 | +0.057* | [+0.015,+0.113] | -0.034 | 0.270 | +0.046 |
| gemma | strategyqa | easy | 54 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| gemma | strategyqa | hard | 24 | -0.021 | 0.083* | 0.076 | +0.007 | [-0.042,+0.062] | +0.021 | 0.083 | +0.007 |
| gemma | truthfulqa | easy | 62 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| llama | aqua | hard | 42 | -0.024 | 0.488* | 0.405 | +0.083* | [+0.034,+0.135] | +0.024 | 0.440 | +0.069 |
| llama | arc | easy | 34 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| llama | arc | hard | 25 | -0.020 | 0.310* | 0.280 | +0.030 | [-0.003,+0.067] | +0.020 | 0.270 | +0.013 |
| llama | commonsenseqa | easy | 41 | -0.012 | 0.012 | 0.008 | +0.004 | [+0.000,+0.012] | +0.012 | 0.012 | +0.004 |
| llama | commonsenseqa | hard | 28 | +0.009 | 0.241* | 0.244 | -0.003 | [-0.045,+0.036] | -0.009 | 0.205 | -0.006 |
| llama | mmlu | hard | 46 | +0.049 | 0.418* | 0.413 | +0.005 | [-0.053,+0.067] | -0.049 | 0.342 | -0.018 |
| llama | strategyqa | easy | 38 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| llama | strategyqa | hard | 19 | -0.092* | 0.118* | 0.101 | +0.018 | [+0.000,+0.044] | +0.092* | 0.118 | +0.018 |
| llama | truthfulqa | easy | 41 | -0.006 | 0.006 | 0.006 | +0.000 | [+0.000,+0.000] | +0.006 | 0.006 | +0.000 |
| llama | truthfulqa | hard | 20 | +0.037 | 0.250* | 0.229 | +0.021 | [-0.013,+0.058] | -0.037 | 0.212 | -0.004 |
| mistral | aqua | hard | 55 | +0.041 | 0.323* | 0.279 | +0.044* | [+0.002,+0.086] | -0.041 | 0.286 | +0.039 |
| mistral | arc | easy | 37 | -0.007 | 0.007 | 0.007 | +0.000 | [+0.000,+0.000] | +0.007 | 0.007 | +0.000 |
| mistral | arc | hard | 23 | +0.000 | 0.207* | 0.188 | +0.018 | [-0.029,+0.062] | +0.000 | 0.196 | +0.014 |
| mistral | commonsenseqa | easy | 49 | -0.005 | 0.005 | 0.012 | -0.007 | [-0.020,+0.000] | +0.005 | 0.005 | -0.007 |
| mistral | commonsenseqa | hard | 20 | -0.050 | 0.125* | 0.100 | +0.025 | [+0.000,+0.075] | +0.050 | 0.113 | +0.025 |
| mistral | mmlu | easy | 16 | -0.016 | 0.016 | 0.016 | +0.000 | [+0.000,+0.000] | +0.016 | 0.016 | +0.000 |
| mistral | mmlu | hard | 36 | -0.007 | 0.236* | 0.229 | +0.007 | [-0.053,+0.065] | +0.007 | 0.201 | -0.002 |
| mistral | strategyqa | easy | 46 | -0.005 | 0.005 | 0.005 | +0.000 | [+0.000,+0.000] | +0.005 | 0.005 | +0.000 |
| mistral | strategyqa | hard | 26 | -0.125* | 0.212* | 0.157 | +0.054* | [+0.000,+0.119] | +0.125* | 0.202 | +0.051 |
| mistral | truthfulqa | easy | 43 | -0.006 | 0.006 | 0.006 | +0.000 | [+0.000,+0.000] | +0.006 | 0.006 | +0.000 |
| mistral | truthfulqa | hard | 29 | +0.009 | 0.198* | 0.161 | +0.037* | [+0.011,+0.069] | -0.009 | 0.181 | +0.032 |
| phi4 | aqua | hard | 47 | -0.021 | 0.559* | 0.528 | +0.030 | [-0.021,+0.082] | +0.021 | 0.495 | +0.039 |
| phi4 | arc | easy | 25 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| phi4 | arc | hard | 32 | +0.047* | 0.281* | 0.305 | -0.023 | [-0.060,+0.008] | -0.047* | 0.258 | -0.029 |
| phi4 | commonsenseqa | easy | 22 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| phi4 | commonsenseqa | hard | 26 | -0.010 | 0.308* | 0.423 | -0.115* | [-0.189,-0.048] | +0.010 | 0.250 | -0.090 |
| phi4 | mmlu | hard | 35 | +0.014 | 0.457* | 0.429 | +0.029 | [-0.033,+0.086] | -0.014 | 0.379 | +0.033 |
| phi4 | strategyqa | easy | 23 | -0.065 | 0.065 | 0.036 | +0.029 | [+0.000,+0.080] | +0.065 | 0.065 | +0.029 |
| phi4 | strategyqa | hard | 23 | -0.022 | 0.196* | 0.232 | -0.036 | [-0.116,+0.043] | +0.022 | 0.196 | -0.036 |
| phi4 | truthfulqa | easy | 32 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| phi4 | truthfulqa | hard | 21 | +0.036 | 0.310* | 0.341 | -0.032 | [-0.107,+0.036] | -0.036 | 0.310 | -0.000 |
| qwen | aqua | easy | 29 | +0.026 | 0.043 | 0.040 | +0.003 | [+0.000,+0.009] | -0.026 | 0.043 | +0.006 |
| qwen | aqua | hard | 39 | -0.019 | 0.391* | 0.372 | +0.019 | [-0.028,+0.066] | +0.019 | 0.340 | +0.015 |
| qwen | arc | easy | 41 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| qwen | arc | hard | 33 | +0.015 | 0.106* | 0.121 | -0.015 | [-0.053,+0.013] | -0.015 | 0.091 | -0.020 |
| qwen | commonsenseqa | easy | 41 | +0.000 | 0.012 | 0.012 | +0.000 | [+0.000,+0.000] | +0.000 | 0.012 | +0.000 |
| qwen | commonsenseqa | hard | 26 | -0.019 | 0.183* | 0.163 | +0.019 | [-0.019,+0.061] | +0.019 | 0.173 | +0.013 |
| qwen | mmlu | easy | 32 | -0.062 | 0.148* | 0.128 | +0.021* | [+0.003,+0.047] | +0.062 | 0.133 | +0.026 |
| qwen | mmlu | hard | 33 | +0.083 | 0.227* | 0.215 | +0.013 | [-0.028,+0.051] | -0.083 | 0.189 | -0.015 |
| qwen | strategyqa | easy | 43 | -0.029 | 0.029 | 0.006 | +0.023 | [+0.000,+0.070] | +0.029 | 0.029 | +0.023 |
| qwen | strategyqa | hard | 34 | -0.007 | 0.176* | 0.132 | +0.044* | [+0.015,+0.083] | +0.007 | 0.176 | +0.044 |
| qwen | truthfulqa | easy | 44 | +0.000 | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | +0.000 | 0.000 | +0.000 |
| qwen | truthfulqa | hard | 30 | -0.008 | 0.192* | 0.175 | +0.017 | [-0.025,+0.064] | +0.008 | 0.175 | +0.022 |
