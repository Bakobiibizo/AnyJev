# Draft issue — not submitted

**Suggested title:** Exploratory result: native vocabulary readout with 1–6 cached tokens for value extraction

Hi! We explored a small extension of AnyJev's forward-only approach: whether a frozen
model's existing vocabulary projection can directly expose a requested source value,
without fitting a separate readout. This is a standalone benchmark, not a proposed change
to the decision-mode contract or public API.

Code, raw replies and reproducibility instructions:
https://github.com/Bakobiibizo/AnyJev/tree/research/native-cached-decoding

## What we tested

- Frozen `Qwen/Qwen2.5-7B-Instruct`, pinned revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, full depth (28 blocks), bfloat16.
- A repository-authored synthetic panel: 1,512 independent requests over 216 source
  variants in 24 scene clusters. Each prompt contains the complete source followed by
  one question about actor, object, location, reporting source, temporal anchor,
  possessor or action. No answer choices or shared conversation history.
- Native greedy argmax after prefill, then single-token KV-cached decoding, with fixed
  caps of 1, 2, 3, 4 and 6 predictions. Native EOS ends early; larger caps retain that reply.
- No fitting, backpropagation, new vocabulary labels, model judge or output repair.

We kept three endpoint measurements separate:

1. Exact literal equality with the authored gold value.
2. Nonempty literal prefix agreement, including exact; strictly partial counts are also saved.
3. Whole-reply equality with the complete authored value after only capitalization,
   one optional leading article and the fixture's predeclared verb aliases.

## Measured results

Source: [`results.json`](https://github.com/Bakobiibizo/AnyJev/blob/research/native-cached-decoding/bench/results_cached_decode/2026-10-01/run01/results.json).
Every column uses the same 1,512 requests.

| Native token cap | Exact literal | Prefix including exact | Complete normalized value |
|---:|---:|---:|---:|
| 1 | 233 (15.41%) | 831 (54.96%) | 307 (20.30%) |
| 2 | 636 (42.06%) | 824 (54.50%) | 886 (58.60%) |
| 3 | 721 (47.69%) | 819 (54.17%) | 936 (61.90%) |
| 4 | 818 (54.10%) | 818 (54.10%) | 1017 (67.26%) |
| 6 | 818 (54.10%) | 818 (54.10%) | 1017 (67.26%) |

Results vary strongly by role: at the four-token cap, complete normalized-value recovery
is 216/216 for reported source, 206/216 for location and 199/216 for actor, but only
1/216 for action under the sealed strict metric.

The experiment directly establishes that additional native greedy tokens improve
complete-value recovery under the sealed scoring rule on this panel. The limitation is
equally important: four/six tokens plateau well below universal recovery, and extra text
can reduce endpoint success. We did not choose an adaptive stopping point or expand
equivalences after seeing outputs.

On the measured GB10 batch-one path, incremental cached steps were approximately
84–85 ms, essentially the same cost as prefill; we make no latency advantage claim.
Detailed timings and conditional step counts are retained in the results.

## Caveats and evidence

This is one model, one finite clustered synthetic panel, not arbitrary-language or
held-out-domain evidence. Some questions leave the intended primary action implicit.
The strict action metric accepts declared verb forms, not nominalizations or extra words;
for example `Restoration of an image.` is not accepted as `restored`. No punctuation or
whitespace stripping was added. Native probability is not calibrated semantic confidence,
and matching the first token of greedy generation is inherent in that decoding rule.

The published package contains exact prompts/IDs, raw token trajectories, all budget
scores, per-step timings, the historical pre-inference seal, export provenance and an
independent replay. Every initial token reproduces its prior native observation.
Small random-engine tests independently check cached decoding against ordinary greedy
continuations. No weights are included. The public portability manifest was generated
later and is explicitly distinguished from the historical preregistration.

Reproduction: [instructions](https://github.com/Bakobiibizo/AnyJev/blob/research/native-cached-decoding/docs/native_values_reproduction.md)
and [full results](https://github.com/Bakobiibizo/AnyJev/blob/research/native-cached-decoding/docs/results_cached_decode.md).

Would this be useful as an optional research benchmark for the project? If so, I would
be happy to discuss the scope before offering a PR. This fork leaves the shipped heads,
backends, public decision API and no-generation decision-mode behavior unchanged.
