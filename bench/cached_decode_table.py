"""Render only audited fixed-budget results; no manually typed bench measurements."""
import json
from pathlib import Path

OUT = Path("bench/results_cached_decode/2026-10-01/run01")


def render():
    """Generate all budgets/roles and actual conditional step latencies from result JSON."""
    r = json.loads((OUT / "results.json").read_text())
    assert json.loads((OUT / "validation.json").read_text())["status"] == "pass"
    print("# Native cached decoding at fixed budgets\n")
    print(f"Source: `{OUT}/results.json`; independent replay: `validation.json`; engine: `engine.json`.\n")
    print("Frozen synthetic panel at b28, unchanged prompts, 1,512 requests; "
          "one greedy cached trajectory each. No fitting or semantic judge.\n")
    print("Budgets count native predictions (including EOS when emitted). EOS ends early; larger caps "
          "retain that reply. Exact/prefix are literal. Semantic means whole-reply equality after "
          "only casefold, one leading a/an/the, and the presealed corpus verb aliases. "
          "No punctuation, whitespace or extra-word removal.\n")
    print("| Budget | Exact literal | Prefix including exact | Strict partial | Complete semantic value | "
          "Median incurred cached decode ms |")
    print("|---:|---:|---:|---:|---:|---:|")
    for s in r["summary"]:
        if s["kind"] == "all":
            scores = [f'{s[k]}/{s["n"]} ({100*s[k]/s["n"]:.2f}%)' for k in
                      ("exact_literal", "prefix_agreement", "strict_partial_prefix", "semantic_value")]
            print(f'| {s["budget"]} | ' + " | ".join(scores) + f' | {s["decode_median_ms"]:.2f} |')
    print("\n## Actual stage timing\n")
    print("CUDA-synchronized forward plus argmax/ID transfer. Incremental counts are conditional on "
          "reaching that step, not all requests. Input setup and scoring excluded; cold first request retained.\n")
    print("| Stage | Observations | Median ms | p95 ms |\n|---|---:|---:|---:|")
    for step, t in r["step_timing"].items():
        label = "Prefill → token 1" if step == "1" else f"Cached step {int(step)-1} → token {step}"
        print(f'| {label} | {t["n"]} | {t["median_ms"]:.2f} | {t["p95_ms"]:.2f} |')
    print("\n## Complete semantic values by role\n")
    print("| Role | Budget 1 | Budget 2 | Budget 3 | Budget 4 | Budget 6 |\n|---|---:|---:|---:|---:|---:|")
    for kind in sorted({s["kind"] for s in r["summary"]} - {"all"}):
        group = [s for s in r["summary"] if s["kind"] == kind]
        print(f"| {kind} | " + " | ".join(f'{s["semantic_value"]}/{s["n"]}' for s in group) + " |")
    print("\nThis run measures native cached decoding, not attention-derived outputs. "
          "The panel's authored values/reference choices and finite clustered scope remain limitations; "
          "these measurements do not certify general semantic extraction. "
          "No favorable budget was selected.\n")
    print("Regenerate: `python -m bench.cached_decode_table`. Protocol: `docs/cached_decode_protocol.md`.")


if __name__ == "__main__":
    render()
