"""Independent raw-token/score/timing replay of the operator-approved cached budget panel."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from bench.value_io import digest, save


def audit(out):
    """Replay frozen rules, IDs, EOS and every budget; no model or semantic interpretation."""
    from transformers import AutoTokenizer, GenerationConfig

    p = json.loads((out / "protocol.json").read_text())
    assert digest(out / "protocol.json") == json.loads((out / "approved.json").read_text())["protocol_sha256"]
    assert digest(out / "targets.json") == p["targets_sha256"]
    assert all(digest(f) == h for f, h in p["code_hashes"].items())
    assert all(digest(f) == h for f, h in p["source_hashes"].items())
    tok = AutoTokenizer.from_pretrained(p["model"], revision=p["revision"], local_files_only=True)
    cfg = GenerationConfig.from_pretrained(p["model"], revision=p["revision"], local_files_only=True)
    eos = {cfg.eos_token_id} if isinstance(cfg.eos_token_id, int) else set(cfg.eos_token_id)
    targets = {r["id"]: r for r in json.loads((out / "targets.json").read_text())}
    rows = [json.loads(s) for s in (out / "observations.jsonl").read_text().splitlines()]
    assert len(targets) == len(rows) == len({r["id"] for r in rows}) == 1512
    def norm(s):
        return re.sub(r"^(?:the|an|a) ", "", s.casefold(), count=1)
    for r in rows:
        target = targets[r["id"]]
        allowed = {norm(target["gold"]), norm(target["semantic_key"])}
        if target["kind"] == "action" and target["semantic_key"] in p["synonyms"]:
            allowed.add(norm(p["synonyms"][target["semantic_key"]]))
        assert sorted(allowed) == target["semantic_forms"]
        steps = r["steps"]
        assert 1 <= len(steps) <= 6 and steps[0]["token_id"] == target["expected_native_id"]
        assert len(steps) == 6 or steps[-1]["token_id"] in eos
        ids = []
        for i, step in enumerate(steps, 1):
            assert step["step"] == i and step["cache_length"] == len(target["input_ids"]) + i - 1
            assert np.isfinite(step["forward_and_selection_ms"]) and step["forward_and_selection_ms"] > 0
            assert step["eos"] == (step["token_id"] in eos)
            assert i == len(steps) or not step["eos"]
            ids.append(step["token_id"])
            # Exclude EOS only; any other special token must remain, not be silently repaired.
            assert step["text"] == tok.decode([tid for tid in ids if tid not in eos], skip_special_tokens=False)
        assert r["prefill_ms"] == steps[0]["forward_and_selection_ms"]
        assert r["total_request_ms"] >= sum(s["forward_and_selection_ms"] for s in steps)
        assert [b["budget"] for b in r["budgets"]] == p["budgets"]
        for b in r["budgets"]:
            actual = steps[:b["budget"]]
            text = actual[-1]["text"]
            exact = text == target["gold"]
            prefix = bool(text) and target["gold"].startswith(text)
            expected = {"exact_literal": exact, "prefix_agreement": prefix,
                        "strict_partial_prefix": prefix and not exact, "semantic_value": norm(text) in allowed}
            assert b["scores"] == expected and b["text"] == text
            assert b["emitted_tokens"] == len(actual) and b["eos"] == actual[-1]["eos"]
            assert b["cached_decode_ms"] == sum(s["forward_and_selection_ms"] for s in actual[1:])
    result = json.loads((out / "results.json").read_text())
    for s in result["summary"]:
        group = [b for r in rows if s["kind"] == "all" or s["kind"] == r["kind"]
                 for b in r["budgets"] if b["budget"] == s["budget"]]
        assert len(group) == s["n"]
        assert all(sum(b["scores"][k] for b in group) == s[k] for k in group[0]["scores"])
        assert s["decode_median_ms"] == float(np.median([b["cached_decode_ms"] for b in group]))
    for i, s in result["step_timing"].items():
        values = [r["steps"][int(i) - 1]["forward_and_selection_ms"] for r in rows if len(r["steps"]) >= int(i)]
        assert s == {"n": len(values), "median_ms": float(np.median(values)),
                     "p95_ms": float(np.percentile(values, 95))}
    assert Path(out / "exit-code.txt").read_text().strip() == "0"
    save(out / "validation.json", {"status": "pass", "requests": len(rows),
         "budget_observations": sum(len(r["budgets"]) for r in rows),
         "prefill_native_parity": "1512/1512 exact recorded token IDs",
         "scope": "Token/score/cache-length/EOS/timing replay; random-model greedy parity tested separately",
         "scope_note": "Native cached decoding only; no attention-based result is validated here",
         "audit_code_sha256": digest(__file__), "results_sha256": digest(out / "results.json"),
         "journal_sha256": digest(out / "observations.jsonl")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out)
