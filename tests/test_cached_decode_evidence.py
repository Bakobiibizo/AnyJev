"""Measured evidence regression: fixed budgets remain distinct; no semantic-rule rescue."""
import json
from pathlib import Path


def test_audited_budget_evidence_preserves_all_metrics_and_actual_timing_counts():
    out = Path(__file__).resolve().parents[1] / "bench/results_cached_decode/2026-10-01/run01"
    result = json.loads((out / "results.json").read_text())
    validation = json.loads((out / "validation.json").read_text())
    assert validation["status"] == "pass" and validation["requests"] == 1512
    all_rows = [r for r in result["summary"] if r["kind"] == "all"]
    assert [r["budget"] for r in all_rows] == [1, 2, 3, 4, 6]
    assert [r["exact_literal"] for r in all_rows] == [233, 636, 721, 818, 818]
    assert [r["semantic_value"] for r in all_rows] == [307, 886, 936, 1017, 1017]
    assert [r["strict_partial_prefix"] for r in all_rows] == [598, 188, 98, 0, 0]
    assert [s["n"] for s in result["step_timing"].values()] == [1512, 1512, 1186, 592, 329, 140]
    assert all(r["n"] == 1512 for r in all_rows)
    assert "no attention-based result" in validation["scope_note"]
