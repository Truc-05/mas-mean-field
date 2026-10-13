# Tier-1 decision churn (ΔY) — the ΔY form of the C1 contrast at breadth

Cells: **20** of 84. Offline from `exp6c_raw_*` dumps, zero generation. Estimators verbatim from `analysis_B_matched_mechanism.py:51-61`; `plurality_prediction` imported from the frozen module.

**SCOPE.** Tier-1 has no copying arm, so this does NOT replicate C1's matched-D mechanism-non-equivalence (that stays with `analysis_B`/`analysis_C` on qwen). It establishes the weaker claim: raising dependence churns decisions while leaving net accuracy flat.

`strict` = tie→-1 (analysis_B convention, **primary**). `tiebreak` = seeded random tie-break (boundary-runner convention).

## Pooled over cells — `strict`

| regime | cells | mean dA | mean ΔY | ΔY sig>0 | mean H_repair | ΔY(shared vs none) | ΔY(indep vs none) |
|---|---|---|---|---|---|---|---|
| easy | 10 | -0.016 | 0.078 | 7/10 | +0.016 | 0.150 | 0.147 |
| hard | 10 | -0.014 | 0.210 | 10/10 | +0.014 | 0.331 | 0.327 |

## Pooled over cells — `tiebreak`

| regime | cells | mean dA | mean ΔY | ΔY sig>0 | mean H_repair | ΔY(shared vs none) | ΔY(indep vs none) |
|---|---|---|---|---|---|---|---|
| easy | 10 | -0.019 | 0.069 | 7/10 | +0.019 | 0.143 | 0.141 |
| hard | 10 | -0.022 | 0.187 | 10/10 | +0.022 | 0.311 | 0.299 |

## Per cell

| model | benchmark | regime | n_q | dA | ΔY | ΔY 95% CI | H_repair | ΔY tiebreak | |dA| vs ΔY |
|---|---|---|---|---|---|---|---|---|---|
| llama | aqua | easy | 40 | -0.056* | 0.194* | [0.116,0.275] | +0.056* | 0.159 | 0.29 |
| llama | arc | easy | 40 | -0.013 | 0.028 | [0.000,0.075] | +0.013 | 0.025 | 0.44 |
| llama | truthfulqa | easy | 40 | +0.000 | 0.047* | [0.009,0.100] | +0.000 | 0.044 | 0.00 |
| qwen | aqua | easy | 40 | -0.003 | 0.125* | [0.059,0.200] | +0.003 | 0.113 | 0.03 |
| qwen | arc | easy | 40 | +0.000 | 0.000 | [0.000,0.000] | +0.000 | 0.000 | nan |
| qwen | bbh | easy | 40 | +0.009 | 0.125* | [0.069,0.188] | -0.009 | 0.113 | 0.07 |
| qwen | commonsenseqa | easy | 40 | -0.013 | 0.044* | [0.009,0.087] | +0.013 | 0.041 | 0.29 |
| qwen | mmlu | easy | 40 | -0.062* | 0.166* | [0.094,0.244] | +0.062* | 0.144 | 0.38 |
| qwen | strategyqa | easy | 40 | -0.025 | 0.050* | [0.009,0.113] | +0.025 | 0.050 | 0.50 |
| qwen | truthfulqa | easy | 40 | +0.000 | 0.000 | [0.000,0.000] | +0.000 | 0.000 | nan |
| llama | aqua | hard | 40 | -0.022 | 0.456* | [0.369,0.541] | +0.022 | 0.422 | 0.05 |
| llama | arc | hard | 40 | -0.044 | 0.225* | [0.147,0.306] | +0.044 | 0.184 | 0.19 |
| llama | truthfulqa | hard | 40 | -0.037* | 0.131* | [0.069,0.203] | +0.037* | 0.122 | 0.29 |
| qwen | aqua | hard | 40 | -0.025 | 0.359* | [0.278,0.447] | +0.025 | 0.309 | 0.07 |
| qwen | arc | hard | 40 | +0.006 | 0.087* | [0.047,0.131] | -0.006 | 0.075 | 0.07 |
| qwen | bbh | hard | 40 | -0.009 | 0.091* | [0.044,0.147] | +0.009 | 0.084 | 0.10 |
| qwen | commonsenseqa | hard | 40 | -0.019 | 0.166* | [0.097,0.244] | +0.019 | 0.159 | 0.11 |
| qwen | mmlu | hard | 40 | +0.044 | 0.287* | [0.194,0.388] | -0.044 | 0.247 | 0.15 |
| qwen | strategyqa | hard | 40 | -0.006 | 0.128* | [0.075,0.191] | +0.006 | 0.128 | 0.05 |
| qwen | truthfulqa | hard | 40 | -0.025 | 0.166* | [0.094,0.244] | +0.025 | 0.141 | 0.15 |
