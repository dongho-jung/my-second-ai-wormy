"""The Node side of the environment, seen from Python.

Each worker is a Node process running a row of worlds. They are driven in
lockstep: every action is written to every worker first, and only then are the
answers read back, so the workers are all computing at the same time instead of
one after another.

The wire format is the one `src/env/worker.js` documents — a four-byte
little-endian length and then that many bytes, all numbers little-endian. The
first frame from each worker is JSON describing the layout of every frame after
it, so nothing here hard-codes a size.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORKER = REPO / "src" / "env" / "worker.js"


def _read_exactly(stream, size: int) -> bytes:
    chunks = []
    left = size
    while left > 0:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError("the worker stopped before it finished a frame")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


def _read_frame(stream) -> bytes:
    return _read_exactly(stream, struct.unpack("<I", _read_exactly(stream, 4))[0])


@dataclass
class Layout:
    envs: int
    agents: int
    vector_size: int
    patch_cells: int
    patch_shape: tuple
    map_cells: int
    map_shape: tuple
    head_sizes: list
    stat_fields: list
    action_bytes: int
    offsets: dict
    frame_bytes: int
    engine_sha256: str
    mod: str
    weapon_ids_at: int | None
    weapon_ids_count: int
    weapon_count: int
    weapons_measured: bool
    frameskip: int
    episode_ticks: int
    maps: int
    stock_maps: int
    done_codes: dict

    @classmethod
    def parse(cls, raw: dict) -> "Layout":
        at = 0
        offsets = {}
        for name in raw["order"]:
            size = raw["bytes"][name]
            offsets[name] = (at, at + size)
            at += size
        return cls(
            envs=raw["envs"],
            agents=raw["agents"],
            vector_size=raw["vectorSize"],
            patch_cells=raw["patchCells"],
            patch_shape=tuple(raw["patchShape"]) if raw["patchShape"] else None,
            map_cells=raw.get("mapCells", 0),
            map_shape=tuple(raw["mapShape"]) if raw.get("mapShape") else None,
            head_sizes=[head["choices"] for head in raw["heads"]],
            stat_fields=raw["statFields"],
            action_bytes=raw["actionBytes"],
            offsets=offsets,
            frame_bytes=at,
            engine_sha256=raw["engineSha256"],
            mod=raw["mod"],
            weapon_ids_at=raw.get("weaponIdsAt"),
            weapon_ids_count=int(raw.get("weaponIdsCount", 0)),
            weapon_count=int(raw.get("weaponCount", 0)),
            weapons_measured=bool(raw.get("weaponsMeasured", True)),
            frameskip=raw["frameskip"],
            episode_ticks=raw["episodeTicks"],
            maps=raw["maps"],
            stock_maps=raw.get("stockMaps", 0),
            # What the `dones` byte means. Three states, not two: an episode
            # here ends on a clock rather than on anything the game did, so its
            # last observation is sent to be valued rather than thrown away.
            done_codes=raw.get("doneCodes", {"ongoing": 0, "first": 1, "last": 2}),
        )


class WorkerPool:
    """Several Node processes of worlds, stepped together."""

    def __init__(self, workers: int, config: dict, node: str = "node"):
        if not WORKER.exists():
            raise FileNotFoundError(f"no environment worker at {WORKER}")
        self.processes = []
        self.layouts = []
        for index in range(workers):
            # Every worker gets its own seed, or they would all play the same
            # match in parallel.
            own = dict(config, seed=int(config.get("seed", 1)) + index * 7919)
            process = subprocess.Popen(
                [node, str(WORKER), json.dumps(own)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                cwd=str(REPO),
                bufsize=0,
            )
            self.processes.append(process)
            self.layouts.append(Layout.parse(json.loads(_read_frame(process.stdout))))
        self.layout = self.layouts[0]
        for other in self.layouts[1:]:
            if other.frame_bytes != self.layout.frame_bytes:
                raise RuntimeError("the workers disagree about the frame layout")
        self.workers = workers
        self.slots = workers * self.layout.envs * self.layout.agents
        self.envs = workers * self.layout.envs
        self._frames = [None] * workers
        # The first observation is already waiting: the worker sends it before
        # it has been asked for anything.
        self._collect()

    def _collect(self):
        for index, process in enumerate(self.processes):
            self._frames[index] = _read_frame(process.stdout)

    def observations(self):
        """Vectors, patches, maps, rewards, dones, restarts and finished-episode stats.

        `restarts` is one byte per worm: it came back from the dead on this
        step, so a policy's memory of it should start over even though the
        match, and its `dones` byte, carry on.
        """
        layout = self.layout
        vectors = np.empty((self.slots, layout.vector_size), dtype=np.float32)
        patches = (
            np.empty((self.slots, layout.patch_cells), dtype=np.uint8)
            if layout.patch_cells
            else None
        )
        maps = (
            np.empty((self.slots, layout.map_cells), dtype=np.uint8)
            if layout.map_cells
            else None
        )
        rewards = np.empty(self.slots, dtype=np.float32)
        dones = np.empty(self.envs, dtype=np.uint8)
        restarts = np.empty(self.slots, dtype=np.uint8)
        stats = np.empty((self.envs, len(layout.stat_fields)), dtype=np.float32)
        per_worker = layout.envs * layout.agents
        for index, frame in enumerate(self._frames):
            # Exactly this block, rather than everything from its start and
            # then a slice. `count=-1` reads to the end of the frame, and numpy
            # refuses when what is left over is not a whole number of elements
            # — which it is not whenever the one-byte `dones` block has a
            # length that does not divide four. The defaults happened to
            # (48 envs, then 12); `--envs 2` did not, and the run died on its
            # first observation.
            def take(name, dtype, frame=frame):
                start, stop = layout.offsets[name]
                return np.frombuffer(
                    frame, dtype=dtype,
                    count=(stop - start) // np.dtype(dtype).itemsize,
                    offset=start,
                )
            agents = slice(index * per_worker, (index + 1) * per_worker)
            envs = slice(index * layout.envs, (index + 1) * layout.envs)
            vectors[agents] = take("vectors", np.float32).reshape(per_worker, -1)
            if patches is not None:
                patches[agents] = take("patches", np.uint8).reshape(per_worker, -1)
            if maps is not None:
                maps[agents] = take("maps", np.uint8).reshape(per_worker, -1)
            rewards[agents] = take("rewards", np.float32)
            dones[envs] = take("dones", np.uint8)
            restarts[agents] = take("restarts", np.uint8)
            stats[envs] = take("stats", np.float32).reshape(layout.envs, -1)
        return vectors, patches, maps, rewards, dones, restarts, stats

    def step(self, heads: np.ndarray):
        """`heads` is (slots, head_count) of uint8 choices."""
        flat = np.ascontiguousarray(heads, dtype=np.uint8).reshape(-1)
        per_worker = self.layout.action_bytes
        size = struct.pack("<I", per_worker)
        for index, process in enumerate(self.processes):
            block = flat[index * per_worker : (index + 1) * per_worker].tobytes()
            process.stdin.write(size + block)
            process.stdin.flush()
        self._collect()

    def close(self):
        for process in self.processes:
            try:
                process.stdin.close()
            except OSError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
