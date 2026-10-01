"""Explicit native cached decoding diagnostic, not decision mode; no fitting, judging or repair.

Inputs are the frozen synthetic value-extraction panel and fixed scoring rules.
Budgets cap native predictions including EOS; larger budgets retain an early-EOS reply.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from anyjev.backends.hf import HFBackend
from bench.run import environment
from bench.tasks.native_values import MODEL, PANEL, REVISION, SYNONYMS
from bench.value_io import digest, read_rows, save

BUDGETS = (1, 2, 3, 4, 6)
LIMIT = 1800


def normalize(text):
    """Only casefold and removal of one optional leading English article; no stripping/repair."""
    text = text.casefold()
    for article in ("the ", "an ", "a "):
        if text.startswith(article):
            return text[len(article):]
    return text


def semantic_forms(row):
    """Finite authored values and frozen explicit action synonyms, not inferred equivalence."""
    values = {normalize(row["semantic_key"]), normalize(row["gold"])}
    if row["kind"] == "action" and row["semantic_key"] in SYNONYMS:
        values.add(normalize(SYNONYMS[row["semantic_key"]]))
    return values


def metrics(text, row):
    exact = text == row["gold"]
    prefix = bool(text) and row["gold"].startswith(text)
    return {"exact_literal": exact, "prefix_agreement": prefix,
            "strict_partial_prefix": prefix and not exact, "semantic_value": normalize(text) in semantic_forms(row)}


def trace(model, tokenizer, ids, maximum=6):
    """One full cached prefill and up to maximum-1 single-token forwards; synchronized path timings."""
    import torch

    device = model.device
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    x = torch.tensor([ids], device=device)
    mask = torch.ones_like(x)
    pos = torch.arange(len(ids), device=device)[None, :]
    cache, tokens, steps = None, [], []
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    with torch.inference_mode():
        for step in range(1, maximum + 1):
            sync()
            start = time.perf_counter()
            out = model(input_ids=x, attention_mask=mask, position_ids=pos, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            chosen = int(out.logits[0, -1].argmax().item())
            sync()
            elapsed = (time.perf_counter() - start) * 1000
            assert cache.get_seq_length() == len(ids) + step - 1
            tokens.append(chosen)
            steps.append({"step": step, "token_id": chosen, "forward_and_selection_ms": elapsed,
                          "cache_length": cache.get_seq_length(), "eos": chosen in eos,
                          "text": tokenizer.decode(tokens, skip_special_tokens=True)})
            if chosen in eos:
                break
            x = torch.tensor([[chosen]], device=device)
            mask = torch.ones((1, len(ids) + step), dtype=torch.long, device=device)
            pos = torch.tensor([[len(ids) + step - 1]], device=device)
    return steps


def prepare(out):
    """Seal source hashes, exact questions, acceptable values and scoring before checkpoint inference."""
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("immutable run: use a new directory")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    targets = read_rows(PANEL)
    assert len(targets) == len({r["id"] for r in targets}) == 1512
    for row in targets:
        assert not row["supplementary"]
        assert row["text"][row["start_char"]:row["end_char"]] == row["gold"]
        assert tok.encode(row["prompt"], add_special_tokens=False) == row["input_ids"]
        assert sorted(semantic_forms(row)) == row["semantic_forms"]
    save(out / "targets.json", targets)
    files = ["bench/cached_decode.py", "tests/test_cached_decode.py", "bench/tasks/native_values.py",
             "anyjev/backends/hf.py", "bench/value_io.py", "bench/run.py"]
    save(out / "protocol.json", {"origin": "Native cached decode panel and fixed normalization",
         "source": str(PANEL), "source_hashes": {str(PANEL): digest(PANEL)},
         "code_hashes": {f: digest(f) for f in files}, "targets_sha256": digest(out / "targets.json"),
         "model": MODEL, "revision": REVISION, "depth": 28, "requests": 1512, "budgets": BUDGETS,
         "process_limit_seconds": LIMIT, "sampling": False, "new_fitted_parameters": 0,
         "batch_size": 1, "dtype": "bfloat16", "cache": True,
         "output_budget": "Predicted native tokens including EOS if emitted; stop early on native EOS only",
         "exact_literal": "Decoded reply excluding special EOS, otherwise unmodified, equals canonical gold",
         "prefix_agreement": "Nonempty literal reply is a character prefix of canonical gold, including exact",
         "strict_partial_prefix": "Prefix agreement excluding exact",
         "semantic_value": "Whole reply equals a presealed complete authored value after casefold "
                           "and one optional leading a/an/the",
         "synonyms": SYNONYMS, "synonym_scope": "Only action values with explicitly declared corpus aliases",
         "forbidden": ["whitespace stripping", "punctuation stripping", "extra-word removal", "new equivalences",
                       "model judge", "learned head", "generation beyond budget or EOS"],
         "timing_scope": "CUDA synchronized forward plus argmax/ID transfer; "
                         "input tensor setup and text/scoring separate",
         "license": "Repo-authored Apache-2.0 panel and cached Apache-2.0 checkpoint; no weights vendored",
         "scope": "Native vocabulary projection and cached decoding only; no attention-based measurement"})


def run(out):
    """Preserve each completed request immediately; never overwrite a partial journal."""
    import torch

    protocol = json.loads((out / "protocol.json").read_text())
    for f, h in {**protocol["code_hashes"], **protocol["source_hashes"]}.items():
        assert digest(f) == h
    assert digest(out / "targets.json") == protocol["targets_sha256"]
    if (out / "observations.jsonl").exists():
        raise FileExistsError("preserve previous invocation")
    torch.set_num_threads(8)
    load = time.perf_counter()
    be = HFBackend(MODEL, revision=REVISION, device="cuda:0", device_map={"": "cuda:0"},
                   dtype="bfloat16", batch_size=1, local_files_only=True)
    be.model.requires_grad_(False)
    devices = sorted({str(p.device) for p in be.model.parameters()})
    assert devices == ["cuda:0"]
    save(out / "engine.json", {"devices": devices, "load_ms": (time.perf_counter() - load) * 1000,
         "environment": environment(phase="native-cached-budgets", dtype="bfloat16", batch_size=1),
         "trainable_parameters": sum(p.numel() for p in be.model.parameters() if p.requires_grad)})
    targets = json.loads((out / "targets.json").read_text())
    started = time.perf_counter()
    with (out / "observations.jsonl").open("x") as journal:
        for n, row in enumerate(targets, 1):
            begin = time.perf_counter()
            steps = trace(be.model, be.tokenizer, row["input_ids"])
            assert steps[0]["token_id"] == row["expected_native_id"], "prefill native parity failed; preserve and stop"
            scores = []
            for budget in BUDGETS:
                current = steps[:budget]
                scores.append({"budget": budget, "text": current[-1]["text"],
                               "emitted_tokens": len(current), "eos": current[-1]["eos"],
                               "scores": metrics(current[-1]["text"], row),
                               "cached_decode_ms": sum(s["forward_and_selection_ms"] for s in current[1:])})
            result = {"id": row["id"], "kind": row["kind"], "steps": steps, "budgets": scores,
                      "prefill_ms": steps[0]["forward_and_selection_ms"],
                      "total_request_ms": (time.perf_counter() - begin) * 1000}
            journal.write(json.dumps(result) + "\n")
            journal.flush()
            if n % 100 == 0 or n == len(targets):
                print(n, "/", len(targets), "elapsed_seconds", time.perf_counter() - started, flush=True)
                save(out / "progress.json", {"completed": n, "total": len(targets)})
    finish(out)


def finish(out):
    """Aggregate actual budgets and actual step timings, without extrapolated optimized costs."""
    records = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    assert len(records) == 1512 and len({r["id"] for r in records}) == 1512
    groups = defaultdict(list)
    timing = defaultdict(list)
    for r in records:
        for step in r["steps"]:
            timing[step["step"]].append(step["forward_and_selection_ms"])
        for b in r["budgets"]:
            groups[b["budget"], "all"].append(b)
            groups[b["budget"], r["kind"]].append(b)
    summary = []
    for (budget, kind), group in sorted(groups.items()):
        summary.append({"budget": budget, "kind": kind, "n": len(group),
            **{key: sum(b["scores"][key] for b in group) for key in group[0]["scores"]},
            "early_eos": sum(b["eos"] and b["emitted_tokens"] < budget for b in group),
            "decode_median_ms": float(np.median([b["cached_decode_ms"] for b in group]))})
    save(out / "results.json", {"origin": "protocol.json + targets.json + observations.jsonl",
         "summary": summary, "step_timing": {str(s): {"n": len(v), "median_ms": float(np.median(v)),
            "p95_ms": float(np.percentile(v, 95))} for s, v in sorted(timing.items())},
         "requests": len(records), "prefill_native_parity": "all 1512 exact token IDs",
         "total_request_median_ms": float(np.median([r["total_request_ms"] for r in records])),
         "protocol_sha256": digest(out / "protocol.json"), "journal_sha256": digest(out / "observations.jsonl")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "finish"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "finish": finish}[args.phase](args.out)
