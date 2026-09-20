"""Recorded play, as something a trainer can hold in memory.

Both the cloner and the reinforcement run read demonstrations, and they have to
agree exactly about the layout or one of them learns from noise. So there is one
reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Demos:
    """Every recording that matches an observation, as one pile of samples."""

    def __init__(self, vectors, patches, maps, heads, sources, meta):
        self.vectors = vectors
        self.patches = patches
        self.maps = maps
        self.heads = heads
        self.sources = sources
        self.meta = meta

    def __len__(self):
        return len(self.vectors)

    @property
    def acting(self) -> int:
        """Frames where the player pressed something. The rest teach stillness."""
        return int((self.heads != 0).any(axis=1).sum())

    def thin_idle(self, max_share: float, seed: int = 0) -> "Demos":
        """Keep the frames where nothing was pressed down to a share of the whole.

        Most of any recording is a person thinking, or waiting to respawn. Left
        in, they are the majority class and anything learned from it learns to
        press nothing.
        """
        idle = (self.heads == 0).all(axis=1)
        idle_count = int(idle.sum())
        acting = len(self) - idle_count
        allowed = int(acting * max_share / max(1e-6, 1 - max_share))
        if idle_count <= allowed or acting == 0:
            return self
        rng = np.random.default_rng(seed)
        keep = np.sort(
            np.concatenate(
                [np.flatnonzero(~idle), rng.choice(np.flatnonzero(idle), allowed, replace=False)]
            )
        )
        return Demos(
            self.vectors[keep],
            self.patches[keep],
            self.maps[keep],
            self.heads[keep],
            self.sources,
            self.meta,
        )


def load(directory: Path, only: str | None = None, expect: dict | None = None) -> Demos | None:
    """Read the recordings in a directory, or None when there are none to read.

    `expect` is the shape a caller already has — a recording taken with a
    different observation is skipped rather than reshaped into nonsense.
    """
    metas = sorted(Path(directory).glob("*.json"))
    if only:
        metas = [path for path in metas if path.stem == only]
    shape = None
    if expect:
        shape = (expect["vectorSize"], expect["patchCells"], expect["mapCells"], expect["heads"])
    first = None
    vectors, patches, maps, heads, sources = [], [], [], [], []
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        binary = meta_path.with_suffix(".bin")
        if not binary.exists():
            continue
        key = (meta["vectorSize"], meta["patchCells"], meta["mapCells"], len(meta["heads"]))
        if shape is None:
            shape = key
        elif key != shape:
            continue
        if first is None:
            first = meta
        size = meta["recordBytes"]
        raw = np.fromfile(binary, dtype=np.uint8)
        count = raw.size // size
        if count == 0:
            continue
        raw = raw[: count * size].reshape(count, size)
        at = meta["vectorSize"] * 4
        vectors.append(raw[:, :at].copy().view(np.float32).reshape(count, -1))
        patches.append(raw[:, at : at + meta["patchCells"]].copy())
        at += meta["patchCells"]
        maps.append(raw[:, at : at + meta["mapCells"]].copy())
        at += meta["mapCells"]
        heads.append(raw[:, at : at + len(meta["heads"])].copy())
        sources.append((meta_path.stem, count))
    if not vectors:
        return None
    return Demos(
        np.concatenate(vectors),
        np.concatenate(patches),
        np.concatenate(maps),
        np.concatenate(heads),
        sources,
        first,
    )
