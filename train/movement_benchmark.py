"""A fixed, isolated movement exam for checkpoints and in-memory policies.

Training randomises maps, spawns and destinations so a policy cannot memorise
one route. That distribution is a poor scoreboard, though: a checkpoint that
happened to receive short goals can collect more arrivals than one that received
hard ones. This module runs one worm and one goal for a fixed clock. Repeating it
with the same seed gives every policy the same map, spawn, latency and target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from workers import WorkerPool


VALIDATION_SEED = 0x51A7E
# Ten routes on each of the 17 room maps. One run of 34 moved by two or three
# routes between checkpoints that were no different, which is as large as the
# improvements it was meant to find; paired over 170, a difference of about
# half a second a route is visible.
DEFAULT_EPISODES = 170
DEFAULT_TICKS = 1800


@dataclass(frozen=True)
class MovementScenarioResult:
    seed: int
    map_index: int
    map_name: str
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    assigned_distance: float
    detour: bool
    reached: int
    seconds: float
    speed: float
    efficiency: float


@dataclass(frozen=True)
class MovementBenchmarkResult:
    scenarios: tuple[MovementScenarioResult, ...]
    episode_ticks: int
    frameskip: int

    @property
    def reached(self) -> int:
        return sum(one.reached for one in self.scenarios)

    @property
    def episodes(self) -> int:
        return len(self.scenarios)

    @property
    def success(self) -> float:
        return self.reached / max(1, self.episodes)

    @property
    def seconds(self) -> float:
        return float(np.mean([one.seconds for one in self.scenarios]))

    @property
    def speed(self) -> float:
        return float(np.mean([one.speed for one in self.scenarios]))

    @property
    def efficiency(self) -> float:
        return float(np.mean([one.efficiency for one in self.scenarios]))

    @property
    def distance(self) -> float:
        return float(np.mean([one.assigned_distance for one in self.scenarios]))

    @property
    def detours(self) -> tuple[MovementScenarioResult, ...]:
        return tuple(one for one in self.scenarios if one.detour)

    @property
    def detour_reached(self) -> int:
        return sum(one.reached for one in self.detours)

    @property
    def detour_success(self) -> float:
        return self.detour_reached / max(1, len(self.detours))

    @property
    def detour_seconds(self) -> float:
        return float(np.mean([one.seconds for one in self.detours])) if self.detours else 0.0

    @property
    def rank(self) -> tuple[int, int, float, float, float]:
        """Checkpoint rank: reliability, detour coverage, speed, then route.

        A faster checkpoint never replaces one that solved more scenarios. If
        two solve the same total, prefer the one that solves more obstructed
        routes before comparing time. Failed scenarios already cost the whole
        fixed horizon in ``seconds``.
        """
        return (self.reached, self.detour_reached, -self.seconds, self.speed, self.efficiency)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict:
        return {
            "episodeTicks": self.episode_ticks,
            "frameskip": self.frameskip,
            "scenarios": [
                {
                    "seed": one.seed,
                    "mapIndex": one.map_index,
                    "map": one.map_name,
                    "start": [one.start_x, one.start_y],
                    "goal": [one.target_x, one.target_y],
                    "distance": one.assigned_distance,
                    "detour": one.detour,
                }
                for one in self.scenarios
            ],
        }

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "benchmarkSuccess": self.success,
            "benchmarkReached": self.reached,
            "benchmarkEpisodes": self.episodes,
            "benchmarkSeconds": self.seconds,
            "benchmarkSpeed": self.speed,
            "benchmarkEfficiency": self.efficiency,
            "benchmarkDistance": self.distance,
            "benchmarkDetourReached": self.detour_reached,
            "benchmarkDetourEpisodes": len(self.detours),
            "benchmarkDetourSuccess": self.detour_success,
            "benchmarkDetourSeconds": self.detour_seconds,
            "benchmarkSuite": self.fingerprint,
        }


def fixed_movement_world(
    world: dict,
    shape: dict,
    *,
    levels: list[str],
    generated_maps: int,
    envs: int,
    seed: int,
    episode_ticks: int,
    end_on_arrival: bool = False,
) -> dict:
    """Freeze curricula and make every episode exactly one isolated task.

    `end_on_arrival` stops an episode as soon as its one goal is reached, which
    is all an exam needs to know. The scenario list does not change with it:
    each world's episodes are the same seeds on the same maps however long
    they last.
    """
    config = dict(world)
    radius = config.get("goalRadiusPx")
    if isinstance(radius, list):
        radius = max(radius)
    config.update(
        agents=1,
        observationFoes=int(shape.get("agents", 1)) - 1,
        envs=envs,
        seed=seed,
        stagger=False,
        opponents=0,
        levelPool=generated_maps,
        levelFiles=levels,
        levelSequence="roundRobin",
        episodeTicks=episode_ticks,
        goals="random",
        goalsPerEpisode=1,
        endOnGoals=end_on_arrival,
        goalRadiusPx=radius,
        goalRadiusMode="steps",
        goalPatience=0,
        decisionsDone=0,
        shapingFullAt=0,
        goalProgressFullAt=0,
        observations=[
            "vector",
            *(["patchBytes"] if shape.get("usePatch", True) else []),
            *(["map"] if shape.get("useMap", False) else []),
        ],
    )
    for name in (
        "goalCurriculum",
        "goalRadiusStart",
        "goalRadiusFullAt",
        "goalPatienceMin",
        "goalPatienceStart",
        "goalDeadlineCurriculum",
    ):
        config.pop(name, None)
    return config


def run_movement_benchmark(
    policy,
    shape: dict,
    world: dict,
    *,
    levels: list[str],
    generated_maps: int = 0,
    episodes: int = DEFAULT_EPISODES,
    workers: int = 2,
    envs: int = 6,
    seed: int = VALIDATION_SEED,
    episode_ticks: int = DEFAULT_TICKS,
    device: torch.device | str = "cpu",
    baseline: str | None = None,
    end_on_arrival: bool = True,
) -> MovementBenchmarkResult:
    """Run one policy over a reproducible suite; actions are greedy and fixed.

    Scenario `k * worlds + i` is world `i`'s `k`-th episode, whatever order the
    episodes finish in. That is the order a suite used to be collected in when
    every episode ran the whole clock, so a suite keeps its routes and its
    fingerprint, and episodes can now end on arrival without a fast policy
    being examined on different routes from a slow one.
    """
    if episodes < 1 or workers < 1 or envs < 1 or episode_ticks < 1:
        raise ValueError("movement benchmark sizes and episode_ticks must be positive")
    if baseline not in (None, "still", "random"):
        raise ValueError(f"unknown movement benchmark baseline {baseline}")
    device = torch.device(device)
    config = fixed_movement_world(
        world,
        shape,
        levels=levels,
        generated_maps=generated_maps,
        envs=envs,
        seed=seed,
        episode_ticks=episode_ticks,
        end_on_arrival=end_on_arrival,
    )
    pool = WorkerPool(workers, config)
    layout = pool.layout
    if layout.vector_size != shape["vectorSize"] or layout.head_sizes != list(shape["headSizes"]):
        pool.close()
        raise RuntimeError(
            "the fixed movement benchmark does not match the policy observation/action layout"
        )
    use_patch = bool(shape.get("usePatch", True)) and layout.patch_cells > 0
    use_map = bool(shape.get("useMap", False)) and layout.map_cells > 0
    fields = {name: index for index, name in enumerate(layout.stat_fields)}
    required = {
        "seed",
        "seedLow",
        "seedHigh",
        "mapIndex",
        "goalsAssigned",
        "goalAssignedDistance",
        "goalDetour",
        "goalStartX",
        "goalStartY",
        "goalTargetX",
        "goalTargetY",
        "goalsReached",
        "goalSeconds",
        "goalSpeed",
        "goalPathEfficiency",
    }
    missing = sorted(required - fields.keys())
    if missing:
        pool.close()
        raise RuntimeError(f"movement benchmark worker is missing stats: {', '.join(missing)}")

    was_training = bool(policy.training) if policy is not None else False
    if policy is not None:
        policy.eval()
        memory = torch.zeros(pool.slots, policy.memory_width, device=device)
    else:
        memory = None
    random = np.random.default_rng(seed ^ 0xA57A)
    vectors, patches, maps, _, _, _, _ = pool.observations()
    next_v = torch.as_tensor(vectors, device=device)
    next_p = torch.as_tensor(patches, device=device) if use_patch else None
    next_m = torch.as_tensor(maps, device=device) if use_map else None
    done = torch.zeros(pool.slots, device=device)
    reset = torch.zeros(pool.slots, device=device)
    DONE_FIRST = layout.done_codes["first"]
    DONE_LAST = layout.done_codes["last"]
    DONE_TERMINAL = layout.done_codes.get("terminal", DONE_LAST)
    found: dict[int, MovementScenarioResult] = {}
    finished = np.zeros(pool.envs, dtype=np.int64)

    generated_count = generated_maps if levels else max(1, generated_maps)

    def map_name(index: int) -> str:
        if index < generated_count:
            return f"generated:{index}"
        stock = index - generated_count
        return Path(levels[stock]).name if 0 <= stock < len(levels) else f"map:{index}"

    try:
        while len(found) < episodes:
            restart = ((done == DONE_FIRST) | (reset > 0)).float()
            if baseline == "still":
                heads = torch.zeros(
                    pool.slots, len(layout.head_sizes), dtype=torch.long, device=device
                )
            elif baseline == "random":
                drawn = np.column_stack(
                    [random.integers(0, size, pool.slots) for size in layout.head_sizes]
                )
                heads = torch.as_tensor(drawn, dtype=torch.long, device=device)
            else:
                with torch.no_grad():
                    logits, _, memory = policy(
                        next_v, next_p, next_m, carried=memory, restart=restart
                    )
                    heads = torch.stack([head.argmax(dim=1) for head in logits], dim=1)

            pool.step(heads.to(torch.uint8).cpu().numpy())
            vectors, patches, maps, _, env_done, restarts, stats = pool.observations()
            next_v = torch.as_tensor(vectors, device=device)
            if use_patch:
                next_p = torch.as_tensor(patches, device=device)
            if use_map:
                next_m = torch.as_tensor(maps, device=device)
            done = torch.as_tensor(env_done.astype(np.float32), device=device)
            reset = torch.as_tensor(restarts.astype(np.float32), device=device)

            for index in np.nonzero((env_done == DONE_LAST) | (env_done == DONE_TERMINAL))[0]:
                order = int(finished[index]) * pool.envs + int(index)
                finished[index] += 1
                if order >= episodes:
                    continue
                row = stats[index]
                assigned = int(round(float(row[fields["goalsAssigned"]])))
                reached = int(round(float(row[fields["goalsReached"]])))
                if assigned != 1 or reached not in (0, 1):
                    raise RuntimeError(
                        "fixed movement benchmark must assign exactly one goal and reach it at most once; "
                        f"got assigned={assigned}, reached={reached}"
                    )
                level_index = int(round(float(row[fields["mapIndex"]])))
                found[order] = (
                    MovementScenarioResult(
                        seed=(
                            int(round(float(row[fields["seedHigh"]]))) << 16
                        ) | int(round(float(row[fields["seedLow"]]))),
                        map_index=level_index,
                        map_name=map_name(level_index),
                        start_x=float(row[fields["goalStartX"]]),
                        start_y=float(row[fields["goalStartY"]]),
                        target_x=float(row[fields["goalTargetX"]]),
                        target_y=float(row[fields["goalTargetY"]]),
                        assigned_distance=float(row[fields["goalAssignedDistance"]]),
                        detour=bool(round(float(row[fields["goalDetour"]]))),
                        reached=reached,
                        seconds=float(row[fields["goalSeconds"]]),
                        speed=float(row[fields["goalSpeed"]]),
                        efficiency=float(row[fields["goalPathEfficiency"]]),
                    )
                )
    finally:
        pool.close()
        if policy is not None and was_training:
            policy.train()

    results = tuple(found[order] for order in range(episodes))
    return MovementBenchmarkResult(results, layout.episode_ticks, layout.frameskip)


def assert_same_scenarios(
    left: MovementBenchmarkResult,
    right: MovementBenchmarkResult,
) -> None:
    """Refuse a comparison unless every paired task is demonstrably identical."""
    if left.episodes != right.episodes:
        raise RuntimeError(
            f"movement benchmark produced {left.episodes} left scenarios and {right.episodes} right"
        )
    for index, (a, b) in enumerate(zip(left.scenarios, right.scenarios)):
        endpoints = (a.start_x, a.start_y, a.target_x, a.target_y)
        other_endpoints = (b.start_x, b.start_y, b.target_x, b.target_y)
        if (
            a.seed != b.seed
            or a.map_index != b.map_index
            or a.map_name != b.map_name
            or any(abs(x - y) > 1e-4 for x, y in zip(endpoints, other_endpoints))
            or abs(a.assigned_distance - b.assigned_distance) > 1e-4
            or a.detour != b.detour
        ):
            raise RuntimeError(
                "movement benchmark scenarios drifted apart at "
                f"{index}: seed/map/endpoints/distance "
                f"{a.seed}/{a.map_name}/{endpoints}/{a.assigned_distance:.4f} against "
                f"{b.seed}/{b.map_name}/{other_endpoints}/{b.assigned_distance:.4f}"
            )
