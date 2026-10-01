# Native cached value extraction: interpretation

Origin: fixed-budget extraction from the frozen synthetic panel at full depth, with
separate literal, prefix and whole-value metrics under presealed normalization.
Numerical source: `results.json`; raw replies and stage timings: `observations.jsonl`.
Historical seals are in `evidence/`; the portability export is recorded in `publication.json`.

All requests were measured. Each initial native token matches the frozen fixture's prior
observation. `validation.json` independently replays tokens, EOS, every budget, fixed
normalization, counts and actual stage timings. CUDA placement and environment are recorded
in `engine.json`. No fitted readout, judge, new equivalence or answer repair was used.

Complete normalized values improve from 307/1512 (20.30%) at one token to 1017/1512
(67.26%) at four and six. Literal equality remains separate: 233/1512 (15.41%) at one,
818/1512 (54.10%) at four/six. The full fixed-budget/role tables are generated in
`docs/results_cached_decode.md` from `results.json`; no favorable budget is selected.

Scores are endpoint whole-reply scores, not whether a value appeared at an earlier step.
Object recovery falls from 144/216 at two tokens to 126/216 at four/six; temporal falls
from 141/216 to 99/216. Extra text can invalidate whole-value equality under the fixed
rules. No per-request stopping rule, punctuation removal or new normalization was added.

Action whole-value recovery is 1/216 at four/six. For example `s00-anchor-action` emits
`Restoration of an image.` at six, whereas accepted surfaces are `restored`/`recovered`.
Nominalizations and extra words were not accepted. On that source the original object
question `What was acted upon?` emits `The incident log.` rather than gold `the image`.
Multiple described predicates and an authored primary-action convention are a confound;
these counts are not a general semantic correctness verdict.

On the measured batch-one GB10 path, cached steps were not appreciably cheaper than
prefill. `results.json` gives prefill median 85.30 ms and actual cached-step medians
83.73–85.32 ms. Counts differ because EOS stops early. Median incurred cached cost is
85.32 ms at budget two and 170.86 ms at four/six; prefill is separate. Input setup and
text scoring are excluded from stage timings but included in whole-request timing.
No optimized latency, unseen-domain performance or calibrated token confidence is inferred.

The six-prediction cap and EOS policy are fixed. The 1800-second stall limit did not end
this run. The public export changes only portability/grouping metadata and descriptive
packaging, not the observed replies, scores, target values or timings. This contribution
is native vocabulary decoding only and does not report attention-derived measurements.
