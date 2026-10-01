# Native cached decoding at fixed budgets

Source: `bench/results_cached_decode/2026-10-01/run01/results.json`; independent replay: `validation.json`; engine: `engine.json`.

Frozen synthetic panel at b28, unchanged prompts, 1,512 requests; one greedy cached trajectory each. No fitting or semantic judge.

Budgets count native predictions (including EOS when emitted). EOS ends early; larger caps retain that reply. Exact/prefix are literal. Semantic means whole-reply equality after only casefold, one leading a/an/the, and the presealed corpus verb aliases. No punctuation, whitespace or extra-word removal.

| Budget | Exact literal | Prefix including exact | Strict partial | Complete semantic value | Median incurred cached decode ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 233/1512 (15.41%) | 831/1512 (54.96%) | 598/1512 (39.55%) | 307/1512 (20.30%) | 0.00 |
| 2 | 636/1512 (42.06%) | 824/1512 (54.50%) | 188/1512 (12.43%) | 886/1512 (58.60%) | 85.32 |
| 3 | 721/1512 (47.69%) | 819/1512 (54.17%) | 98/1512 (6.48%) | 936/1512 (61.90%) | 169.07 |
| 4 | 818/1512 (54.10%) | 818/1512 (54.10%) | 0/1512 (0.00%) | 1017/1512 (67.26%) | 170.86 |
| 6 | 818/1512 (54.10%) | 818/1512 (54.10%) | 0/1512 (0.00%) | 1017/1512 (67.26%) | 170.86 |

## Actual stage timing

CUDA-synchronized forward plus argmax/ID transfer. Incremental counts are conditional on reaching that step, not all requests. Input setup and scoring excluded; cold first request retained.

| Stage | Observations | Median ms | p95 ms |
|---|---:|---:|---:|
| Prefill → token 1 | 1512 | 85.30 | 87.67 |
| Cached step 1 → token 2 | 1512 | 85.32 | 88.77 |
| Cached step 2 → token 3 | 1186 | 84.38 | 87.88 |
| Cached step 3 → token 4 | 592 | 84.16 | 87.60 |
| Cached step 4 → token 5 | 329 | 83.94 | 87.34 |
| Cached step 5 → token 6 | 140 | 83.73 | 87.23 |

## Complete semantic values by role

| Role | Budget 1 | Budget 2 | Budget 3 | Budget 4 | Budget 6 |
|---|---:|---:|---:|---:|---:|
| action | 0/216 | 2/216 | 1/216 | 1/216 | 1/216 |
| actor | 45/216 | 152/216 | 191/216 | 199/216 | 199/216 |
| location | 0/216 | 110/216 | 139/216 | 206/216 | 206/216 |
| object | 71/216 | 144/216 | 138/216 | 126/216 | 126/216 |
| possessive | 57/216 | 137/216 | 147/216 | 170/216 | 170/216 |
| reported_source | 72/216 | 200/216 | 216/216 | 216/216 | 216/216 |
| temporal | 62/216 | 141/216 | 104/216 | 99/216 | 99/216 |

This run measures native cached decoding, not attention-derived outputs. The panel's authored values/reference choices and finite clustered scope remain limitations; these measurements do not certify general semantic extraction. No favorable budget was selected.

Regenerate: `python -m bench.cached_decode_table`. Protocol: `docs/cached_decode_protocol.md`.
