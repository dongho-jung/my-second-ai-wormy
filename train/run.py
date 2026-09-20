"""Writing down how a run is going, in the format the monitor reads.

The same two files `src/train/recorder.js` writes: `run.json` for what was
decided once and an append-only `metrics.jsonl` for one line per measurement.
Append-only because a run that is killed halfway should still be readable, and
because the monitor follows it by reading whatever bytes have appeared since it
last looked.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO / "artifacts" / "runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_run_id() -> str:
    stamp = _now().replace(":", "-").replace(".", "-").replace("Z", "")
    return f"{stamp}-{secrets.token_hex(2)}"


class Run:
    def __init__(self, label: str = None, meta: dict = None, directory: Path = None):
        self.id = new_run_id()
        self.path = Path(directory or DEFAULT_RUNS) / self.id
        self.path.mkdir(parents=True, exist_ok=True)
        self.run = {
            "schemaVersion": 1,
            "id": self.id,
            "label": label,
            "startedAt": _now(),
            "endedAt": None,
            "status": "running",
            "meta": meta or {},
        }
        self._describe()
        self._metrics = open(self.path / "metrics.jsonl", "a", buffering=1)

    def _describe(self):
        (self.path / "run.json").write_text(json.dumps(self.run, indent=2) + "\n")

    def record(self, **fields):
        self._metrics.write(json.dumps({"at": _now(), **fields}, default=float) + "\n")
        # Flushed every line: the monitor is reading this file as it is written.
        self._metrics.flush()
        os.fsync(self._metrics.fileno())

    def note(self, text: str, **fields):
        self.record(note=text, **fields)

    def close(self, status: str = "done", **fields):
        self.run["status"] = status
        self.run["endedAt"] = _now()
        self.run["meta"].update(fields)
        self._describe()
        self._metrics.close()
