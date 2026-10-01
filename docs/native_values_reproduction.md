# Reproduce the native value-decoding results

This contribution is experimental benchmark code and measured synthetic fixtures only.
AnyJev's production APIs, calibration methods, shipped heads and existing backend are unchanged.

## Install and verify the published evidence (no GPU/model weights required)

From the repository root, with Python 3.10+:

```bash
python -m pip install -e '.[dev,hf]' accelerate
ruff check anyjev bench demo scripts space tests
pytest -q
python -m bench.cached_decode_audit --out bench/results_cached_decode/2026-10-01/run01
python -m bench.cached_decode_table > docs/results_cached_decode.md
```

The audit needs the pinned tokenizer/config files; it does not load model weights or run
inference. The loader is local-files-only. If these files are not cached, fetch the pinned
checkpoint's small config/tokenizer files first. Reuse an existing model cache if present;
do not make another model-weight copy. CPU-only core tests use numpy and skip the optional
engine test.

Optional small random-model cache-parity check:

```bash
ANYJEV_CACHED_ENGINE=1 pytest -q tests/test_cached_decode.py
```

This verifies cache mechanics, not checkpoint semantic capability.

## Rerun model inference into a NEW directory

A CUDA device with sufficient space for the bfloat16 checkpoint is required. Measurements
used an NVIDIA GB10 at batch one. `engine.json` records torch/transformers versions and
hardware output; different hardware/software can change timing and numerical details.
The pinned checkpoint is Apache-2.0 and is not vendored. Cache that exact revision once
using your normal Hugging Face tooling, then run from the repository root:

```bash
OUT=bench/results_cached_decode/$(date +%F)/reproduction01
python -m bench.cached_decode prepare --out "$OUT"
# For a fresh run, create the audit approval pointer after preparation and before inference:
python - "$OUT" <<'PY'
import json, pathlib, sys
from bench.value_io import digest, save
p = pathlib.Path(sys.argv[1])
save(p / 'approved.json', {'origin': 'Fresh frozen native-value benchmark',
     'protocol_sha256': digest(p / 'protocol.json'),
     'targets_sha256': digest(p / 'targets.json'), 'inference_started': False})
PY
timeout --kill-after=30s 1800s python -u -m bench.cached_decode run --out "$OUT" > "$OUT/process.txt" 2>&1
RC=$?; printf '%s\n' "$RC" > "$OUT/exit-code.txt"
python -m bench.cached_decode_audit --out "$OUT"
```

If a process fails or times out, preserve that directory and journal. Do not use repeated
runs to erase a failure. The original measured directory refuses overwrites. The code
asserts CUDA placement, zero trainable parameters and parity with the fixture's first
native token on every request. It does not alter prompts, fit a head or invoke a judge.

## Read the artifacts

- `targets.json`: exact prompts, gold values, tokenizer IDs and frozen acceptable forms.
- `observations.jsonl`: every native output token, reply after each step, EOS/cache checks,
  all budget scores, per-step timing and whole-request timing.
- `results.json`: every budget/role and median/p95 stage timing with reached-step counts.
- `engine.json`: actual CUDA placement and measurement environment.
- `validation.json`: independent replay verdict and primary hashes.
- `publication.json`: provenance of the portability export; original reply/timing journal
  retained byte for byte. Historical protocol/results/audit are under `evidence/`.

No weights, private local state or unrelated experimental history are part of this package.
The issue/PR drafts under `docs/contribution/` are text only; neither has been submitted.
