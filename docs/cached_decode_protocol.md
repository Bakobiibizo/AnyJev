# Native value extraction with fixed cached-decode budgets

A standalone experimental benchmark, not a change to AnyJev's public decision API.
It asks one value question per independent prompt and reads the pretrained model's
native vocabulary projection. No fitted readout, model judge or output repair.

## Inputs

`bench/tasks/native_values.jsonl.gz` is the frozen repository-authored synthetic panel:
1,512 requests over 216 source variants in 24 scene clusters. Seven questions concern
actor, object, location, reporting source, temporal anchor, possessor and action. Records
include exact source text, prompt, token IDs, literal offsets, canonical value, declared
acceptable surfaces and the previously observed first native token. No supplementary
examples are included. These are finite fixtures, not a general-language evaluation.

Model: `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Full depth (28 blocks), bfloat16, batch one,
eval/inference mode, zero trainable parameters. No weights are included in the repository.

System: `Answer the question using the state. Give the shortest answer, without explanation.`
User: complete `State:\n{source}\n\nQuestion: {question}`, followed by the model's native
assistant-response header. No answer choices or prior question/reply history.

## Execution and metrics

One cached prefill emits token one. Each subsequent call consumes exactly one predicted
token and reuses its KV cache. Greedy native argmax, no sampling or repetition penalty.
Record a single trajectory up to six native predictions and report budgets 1, 2, 3, 4, 6.
EOS ends a request early, counts as a prediction, and is excluded from text; larger caps
retain the finished reply and actual cost. First-token IDs must reproduce the frozen
fixture's prior native IDs. An external 1,800-second limit plus 30-second kill grace
protects against stalls; preserve partial output rather than rerun into the same directory.

Three separate endpoint measurements:

1. Exact literal: complete decoded reply equals canonical gold without modification.
2. Literal prefix agreement: nonempty reply prefixes canonical gold, including exact.
   Strictly partial prefix counts are also stored separately.
3. Complete semantic value: the WHOLE reply equals a frozen complete authored value after
   casefold and removal of at most one leading `a `, `an ` or `the `. Only action values
   receive the fixture's explicit verb aliases. No whitespace/punctuation stripping,
   nominalization equivalence, extra-word removal, substring matching or semantic judge.

Additional text can therefore reduce endpoint success. Do not choose a per-request
stopping point after observing outputs or extend equivalences to improve scores.

CUDA synchronization bounds each forward plus argmax/ID transfer. Prefill and each actual
cached step are separate; later-step counts are conditional on reaching that step. Input
setup and decoding/scoring are outside those stage timings and inside whole-request timing.
No optimized latency is inferred by subtracting instrumentation.

## Evidence and portability

Measured outputs: `bench/results_cached_decode/2026-10-01/run01/`. Its
`evidence/original-protocol.json` preserves the historical pre-inference seal;
`publication.json` records the public portability export. Public source/manifest hashes
were regenerated after measurement; this is explicitly NOT a new preregistration.
Only grouping metadata and dependency/provenance packaging changed. All numerical scores,
reply tokens and actual timings remain unchanged; the original journal hash is verified.

`bench/cached_decode_audit.py` independently replays every token, budget, fixed metric,
EOS, cache length, source hash and timing aggregation without loading model weights.
Analytical tests and a small random-engine cached-versus-greedy parity test cover code.
These checks do not establish general semantic correctness or calibrated confidence.

Reproduction instructions: `docs/native_values_reproduction.md`.
