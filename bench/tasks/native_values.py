"""Frozen Apache-2.0 synthetic value-extraction panel; no external data or teacher labels.

The compressed input fixture is the exact 1,512-request authored panel used for the
reported measurements. Rendering order, literal offsets and tokenizer IDs are preserved.
Checkpoint license: Apache-2.0 at the pinned revision; see THIRD_PARTY.md. No weights vendored.
"""
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PANEL = Path("bench/tasks/native_values.jsonl.gz")
SYNONYMS = {"restarted": "rebooted", "inspected": "examined", "repaired": "fixed", "replaced": "substituted",
            "moved": "relocated", "copied": "duplicated", "stopped": "halted", "started": "launched",
            "deployed": "rolled out", "updated": "revised", "archived": "filed away", "restored": "recovered"}
