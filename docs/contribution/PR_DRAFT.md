# Draft PR — not submitted

**Suggested title:** Add an experimental native cached-decoding value-extraction benchmark

## Summary

Adds a standalone benchmark of a frozen model's native vocabulary projection at fixed
1/2/3/4/6-token cached decode caps. No learned head, judge, output repair or public API
change. Explicit generation is confined to this experimental benchmark, not decision mode.

Fork branch: https://github.com/Bakobiibizo/AnyJev/tree/research/native-cached-decoding
Base: upstream `main` at `10d5db91dda38dbde74c6abc1c075ce6463723d1`.

## Included

- Frozen Apache-2.0 synthetic panel with exact prompts, IDs and canonical/acceptable values.
- Native cached runner, separate prefill/incremental timing and early EOS handling.
- Fixed exact/prefix/whole-value normalization and independent artifact replay.
- Measured raw trajectories, every budget and role, hardware/software metadata and hashes.
- Historical pre-inference seal plus explicit public-export provenance.
- Analytical tests, opt-in small random-engine cache parity, and evidence regression tests.
- Reproduction guide, generated report and draft issue text; no weights or unrelated history.

## Result

From `bench/results_cached_decode/2026-10-01/run01/results.json`, on 1,512 requests:
complete normalized value recovery is 307/1512 (20.30%) at one token and
1017/1512 (67.26%) at four/six. Exact literal equality is respectively
233/1512 (15.41%) and 818/1512 (54.10%). All budgets and strictly partial prefix counts
are retained, rather than selecting a favorable budget.

Median prefill is 85.30 ms; actual cached-step medians are 83.73–85.32 ms on batch-one
GB10, so this does not claim cheap incremental decoding relative to prefill. See the
JSON and generated report for sample counts, p95 and conditional timing scope.

## Limitations

One model and finite clustered synthetic fixtures. Intended primary-action/reference
choices can be ambiguous. Normalization permits only the frozen capitalization/article/
verb-alias rules, not punctuation removal, nominalization or extra-word repair. Extra
text can reduce endpoint success. No general semantic accuracy or calibrated confidence
claim; no change to AnyJev's production decision contract.

## Validation

Run the repository lint/test commands and the independent replay in
`docs/native_values_reproduction.md`. The runner asserts full CUDA placement, frozen
parameters and parity of every initial token with its recorded native observation.

This text is only a local draft. No upstream issue or PR has been created.
