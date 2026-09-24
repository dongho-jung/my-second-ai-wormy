"""A genetic algorithm over the movement policy's weights, from scratch or from a checkpoint.

No gradients. A population of whole networks is scored on the same tasks, the
best few become parents, and the next generation is the best one kept as it
is plus mutated copies of the parents: every weight moved by Gaussian noise
of size --sigma. This is the "deep neuroevolution" recipe (Such et al., 2017):
truncation selection, elitism, mutation only.

Every member of a generation meets exactly the same tasks: the worlds replay
one scenario per member (`replicas` in src/env/vec.js), and the whole
population decides in one batched pass (train/population.py). New tasks are
drawn every generation, so no set of routes can be learned by heart.

A task costs its seconds as a share of the clock when it is reached, and one
plus the share of the distance still left at the closest point when it is not,
so a population that reaches nothing yet is still ranked by who came nearest.

From scratch the destinations start close and the clock short. When the
elite reaches --promote-at of a generation's tasks the radius grows by a fifth,
and it shrinks by a tenth below --demote-at, up to the full --goal-radius; the
clock follows the radius. The tasks mix everything a worm has to do to get
somewhere: walking, jumping and the rope for destinations above
(--goal-above), going around what blocks the straight line (--goal-detour),
and digging into destinations packed in dirt (--goal-dig).

Every --check-every generations the elite takes the paired exam at the full
radius and clock (train/stability.py), and best.pt moves only when it beats
the champion there. Checkpoints, run.json and metrics.jsonl are the trainer's
own, so the monitor, Watch and `npm run evaluate` read an evolved policy like
any other.

    npm run train -- --algorithm genetic                      # from scratch
    npm run train -- --algorithm genetic --resume <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import experiment_champion, experiment_checkpoint, file_digest
from movement_benchmark import (
    DEFAULT_EPISODES as DEFAULT_BENCHMARK_EPISODES,
    DEFAULT_TICKS,
    fixed_movement_world,
    run_movement_benchmark,
)
from policy import MAP_CHANNELS, PATCH_CHANNELS, WormPolicy, policy_from_shape
from population import Population
from run import Run
from stability import MovementGuard, Routes, compare_routes, score_suite
from workers import REPO, WorkerPool

# How the clock follows the radius: long enough to walk and climb there, never
# more than the exam's thirty seconds.
CLOCK_BASE_SECONDS = 6.0
CLOCK_SECONDS_PER_PX = 1 / 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", default="genetic", choices=["genetic"],
                        help="kept so one command line can name the trainer")
    parser.add_argument("--resume", default=None,
                        help="a checkpoint, a run's directory, or with --experiment-id the runs directory; "
                             "from scratch when there is nothing to carry on from")
    parser.add_argument("--experiment-id", default=None)
    world = parser.add_argument_group("the world, from scratch (a checkpoint brings its own)")
    world.add_argument("--agents", type=int, default=6,
                       help="worms the observation has room for; one moves, the rest of the slots stay empty")
    world.add_argument("--patch-scale", type=int, default=4)
    world.add_argument("--stock-levels", type=int, default=17)
    world.add_argument("--goal-radius", type=float, default=1600, help="the radius the curriculum ends at")
    world.add_argument("--goal-radius-start", type=float, default=96)
    world.add_argument("--goal-above", type=float, default=0.4, help="share of destinations a jump cannot reach")
    world.add_argument("--goal-detour", type=float, default=0.4, help="share with ground across the straight line")
    world.add_argument("--goal-dig", type=float, default=0.25, help="share buried in dirt, reached only by digging")
    world.add_argument("--hook-rays", type=int, default=12,
                       help="directions the vector reports where a rope would hook; 0 leaves them out")
    evolution = parser.add_argument_group("evolution")
    evolution.add_argument("--population", type=int, default=128, help="networks scored each generation")
    evolution.add_argument("--parents", type=int, default=13, help="the best this many breed the next generation")
    evolution.add_argument("--sigma", type=float, default=0.004,
                           help="standard deviation of the noise added to every weight of a child")
    evolution.add_argument("--tasks", type=int, default=16,
                           help="tasks every member is scored on each generation (same tasks for all)")
    evolution.add_argument("--promote-at", type=float, default=0.8)
    evolution.add_argument("--demote-at", type=float, default=0.4)
    evolution.add_argument("--generations", type=int, default=0, help="stop after this many; 0 runs on")
    evolution.add_argument("--workers", type=int, default=6,
                           help="worker processes for each half of the population; the halves take turns")
    evolution.add_argument("--scenarios-per-worker", type=int, default=1,
                           help="distinct tasks a worker plays at once; each is replayed for every member")
    exam = parser.add_argument_group("the paired exam that moves best.pt")
    exam.add_argument("--check-every", type=int, default=10, help="generations between exams of the elite")
    exam.add_argument("--benchmark-episodes", type=int, default=DEFAULT_BENCHMARK_EPISODES)
    exam.add_argument("--benchmark-seed", type=int, default=334462)
    exam.add_argument("--validation-seeds", default="1334462")
    exam.add_argument("--benchmark-ticks", type=int, default=DEFAULT_TICKS)
    exam.add_argument("--benchmark-workers", type=int, default=8)
    exam.add_argument("--benchmark-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument("--label", default=None)
    args, unknown = parser.parse_known_args(argv)
    # The launcher passes the deployment's shared flags through; what is not
    # ours is named rather than silently dropped.
    if unknown:
        print(f"evolve | ignoring flags that belong to another trainer: {' '.join(unknown)}", flush=True)
    if not 1 <= args.parents <= args.population or args.population < 2:
        parser.error("--population must be at least 2 and --parents between 1 and --population")
    if args.sigma <= 0 or args.tasks < 1 or args.check_every < 1:
        parser.error("--sigma, --tasks and --check-every must be positive")
    return args


def fresh_world(args):
    levels = sorted(
        str(path) for path in (REPO / "artifacts" / "maps" / "dsds-cs").glob("*")
        if path.suffix.lower() in (".lev", ".png")
    )[: args.stock_levels]
    if not levels:
        raise SystemExit("no room maps to evolve on: run `npm run maps`")
    return dict(
        agents=args.agents, episodeTicks=3600, frameskip=4, patchScale=args.patch_scale,
        inputLatencyTicks=[6, 21], levelPool=0, levelFiles=levels, levelOptions={"width": 504},
        weaponPool="room", loadout="random", shapingFullAt=0, shapingFloor=0, banStart=[],
        rules={"bonusDrops": 3, "bonusSpawnTicks": 480, "weaponChangeDelay": 45},
        observations=["vector", "patchBytes", "map"], goals="random", weights="movement",
        lockWeapons=True, goalRadiusPx=float(args.goal_radius), goalPatience=450, goalsPerEpisode=1,
        goalProgressMode="best", goalDetourShare=args.goal_detour, goalAboveShare=args.goal_above,
        goalAbovePx=48, goalDigShare=args.goal_dig, goalRadiusMode="steps", hookRays=args.hook_rays,
    )


def locate(args):
    """What to carry on from, and the champion a restarted experiment was keeping."""
    if not args.resume:
        return None, None
    resume = Path(args.resume)
    if args.experiment_id and resume.is_dir():
        return experiment_checkpoint(resume, args.experiment_id), experiment_champion(resume, args.experiment_id)
    if resume.is_file():
        return resume, None
    for name in ("best.pt", "policy.pt"):
        if (resume / name).is_file():
            return resume / name, None
    return None, None


def save(path, policy, layout, step, **extra):
    """Written beside and renamed over, so a reader never sees half a checkpoint."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save({"policy": policy.state_dict(), "layout": layout, "step": step, **extra}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate(template, world, shape_agents, workers, decisions=120, seed=7):
    """The input normaliser, measured on worlds driven by random keys."""
    config = fixed_movement_world(world, {"agents": shape_agents, "usePatch": True, "useMap": True}, levels=world["levelFiles"],
                                  generated_maps=0, envs=16, seed=seed, episode_ticks=decisions * 4 + 4)
    pool = WorkerPool(workers, config)
    rng = np.random.default_rng(seed)
    try:
        for _ in range(decisions):
            vectors, _, _, _, _, _, _ = pool.observations()
            template.norm.observe(torch.as_tensor(vectors, dtype=torch.float32))
            heads = np.column_stack([rng.integers(0, size, pool.slots) for size in pool.layout.head_sizes])
            pool.step(heads.astype(np.uint8))
    finally:
        pool.close()


def score_population(population, world, layout, *, seed, tasks, workers, per_worker, ticks, groups=2):
    """Every member on the same tasks: cost, reached share, buried-task reached share and decisions.

    The population is split into `groups`, each with its own pool of workers
    on the same seed and settings, so they play the same scenarios. While one
    pool simulates, the network decides for the next: the two halves of a
    decision that used to take turns now overlap.

    Inside a pool, world i of a worker plays scenario floor(i / G) and belongs
    to the group's member i % G, so the replicas of a scenario are one per
    member. Scenario (worker w, slot s, episode k) is the same for everybody
    whenever it finishes; results are kept by that key, so a member that
    arrives early is not scored on different tasks from one that does not.
    """
    members = population.size
    groups = max(1, min(groups, members))
    bounds = np.linspace(0, members, groups + 1).astype(int)
    parts = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
    scenarios = workers * per_worker
    rounds = -(-tasks // scenarios)
    clock = ticks / 60
    cost = np.full((members, scenarios * rounds), np.nan)
    reached = np.zeros((members, scenarios * rounds))
    digs = np.zeros(scenarios * rounds, dtype=bool)
    template = population.template
    states = []
    try:
        for part in parts:
            size = part.stop - part.start
            envs = per_worker * size
            config = fixed_movement_world(
                world, layout, levels=world.get("levelFiles") or [], generated_maps=world.get("levelPool", 0),
                envs=envs, seed=seed, episode_ticks=ticks, end_on_arrival=True,
            )
            config["replicas"] = size
            pool = WorkerPool(workers, config)
            states.append({"pool": pool, "part": part, "size": size, "envs": envs, "pending": False})
            shape = pool.layout
            if shape.vector_size != layout["vectorSize"] or shape.head_sizes != list(layout["headSizes"]):
                raise RuntimeError("the evolution worlds do not match the policy's observation or actions")
            worlds = pool.envs
            # World w*envs + s*size + m -> the group's member m; its j-th world is (w, s).
            order = torch.arange(worlds).view(workers, per_worker, size).permute(2, 0, 1).reshape(size, -1)
            states[-1].update(
                order=order, worlds=worlds, finished=np.zeros(worlds, dtype=np.int64),
                memory=torch.zeros(size, worlds // size, template.memory_width),
                fields={name: index for index, name in enumerate(shape.stat_fields)},
                first=shape.done_codes["first"],
                closing=(shape.done_codes["last"], shape.done_codes.get("terminal", shape.done_codes["last"])),
                frame=pool.observations(), done=np.zeros(worlds), restarts=np.zeros(worlds),
            )
        decisions = 0

        def decide(state):
            vectors, patches, maps = state["frame"][:3]
            order = state["order"]
            v = torch.as_tensor(vectors)[order]
            p = torch.as_tensor(patches)[order] if template.use_patch else None
            m = torch.as_tensor(maps)[order] if template.use_map else None
            restart = torch.as_tensor(
                ((state["done"] == state["first"]) | (state["restarts"] > 0)).astype(np.float32))[order]
            heads, state["memory"] = population.act(v, p, m, state["memory"], restart, part=state["part"])
            flat = torch.zeros(state["worlds"], heads.shape[-1], dtype=torch.long)
            flat[order.reshape(-1)] = heads.reshape(-1, heads.shape[-1])
            state["pool"].send(flat.to(torch.uint8).numpy())
            state["pending"] = True

        def complete(state):
            return not np.isnan(cost[state["part"]]).any()

        for state in states:
            decide(state)
        while any(state["pending"] for state in states):
            for state in states:
                if not state["pending"]:
                    continue
                state["pool"].receive()
                state["pending"] = False
                frame = state["pool"].observations()
                state["frame"] = frame
                env_done, restarts, stats = frame[4], frame[5], frame[6]
                state["done"], state["restarts"] = env_done, restarts
                decisions += state["worlds"]
                fields, size, envs = state["fields"], state["size"], state["envs"]
                for world_index in np.nonzero(np.isin(env_done, state["closing"]))[0]:
                    episode = int(state["finished"][world_index])
                    state["finished"][world_index] += 1
                    if episode >= rounds:
                        continue
                    worker, rest = divmod(int(world_index), envs)
                    slot, member = divmod(rest, size)
                    task = episode * scenarios + worker * per_worker + slot
                    row = stats[world_index]
                    hit = float(row[fields["goalsReached"]]) > 0.5
                    member += state["part"].start
                    reached[member, task] = hit
                    digs[task] = float(row[fields["goalDig"]]) > 0.5
                    cost[member, task] = (
                        float(row[fields["goalSeconds"]]) / clock if hit
                        else 1.0 + float(row[fields["goalClosestShare"]])
                    )
                if not complete(state):
                    decide(state)
    finally:
        for state in states:
            state["pool"].close()
    take = slice(0, tasks)
    dig_tasks = digs[take]
    dig_reached = reached[:, take][:, dig_tasks].mean(axis=1) if dig_tasks.any() else np.full(members, np.nan)
    return cost[:, take].mean(axis=1), reached[:, take].mean(axis=1), dig_reached, int(dig_tasks.sum()), decisions


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    generator = torch.Generator().manual_seed(args.seed)
    start_from, champion_from = locate(args)
    carried = torch.load(start_from, map_location="cpu", weights_only=False) if start_from else None
    if carried is not None:
        layout = carried["layout"]
        world = dict(layout.get("world") or {})
        template = policy_from_shape(layout)
        template.load_state_dict(carried["policy"])
        step = int(carried.get("step", 0))
        radius = float(carried.get("goalRadius") or args.goal_radius)
        generation = int(carried.get("generation", 0))
    else:
        world = fresh_world(args)
        probe = WorkerPool(1, fixed_movement_world(world, {"agents": args.agents, "usePatch": True, "useMap": True}, levels=world["levelFiles"],
                                                   generated_maps=0, envs=1, seed=1, episode_ticks=60))
        shape = probe.layout
        probe.close()
        template = WormPolicy(
            shape.vector_size, shape.head_sizes, patch_shape=tuple(shape.patch_shape[1:]), use_patch=True,
            use_map=True, map_side=shape.map_shape[1], weapon_ids_at=shape.weapon_ids_at,
            weapon_ids_count=shape.weapon_ids_count, weapon_count=shape.weapon_count, conv_padding=True,
        )
        layout = {
            "vectorSize": shape.vector_size, "headSizes": shape.head_sizes, "world": world,
            "usePatch": True, "patchShape": list(shape.patch_shape[1:]), "weaponIdsAt": shape.weapon_ids_at,
            "weaponIdsCount": shape.weapon_ids_count, "weaponCount": shape.weapon_count, "useMap": True,
            "mapSide": shape.map_shape[1], "convPadding": True, "patchChannels": PATCH_CHANNELS,
            "mapChannels": MAP_CHANNELS, "agents": args.agents, "frameskip": 4, "episodeTicks": 3600,
        }
        print("evolve | measuring the input normaliser on random keys", flush=True)
        calibrate(template, world, args.agents, args.workers)
        step = 0
        radius = float(args.goal_radius_start)
        generation = 0
    template.eval()
    full_radius = float(world.get("goalRadiusPx") or args.goal_radius)
    if carried is None:
        # Every member its own random network: nothing is known yet, so the
        # first generation is as varied as the population is large.
        members = []
        for index in range(args.population):
            torch.manual_seed(args.seed * 100_003 + index)
            one = WormPolicy(
                template.norm.mean.shape[0], template.head_sizes, patch_shape=template.patch_shape,
                use_map=template.use_map, map_side=template.map_side, weapon_ids_at=template.weapon_ids_at,
                weapon_ids_count=template.weapon_ids_count, weapon_count=template.weapon_count,
                conv_padding=template.conv_padding,
            )
            members.append(one.state_dict())
        population = Population(template, members)
    else:
        population = Population(template, count=args.population)
        population.breed([0], elite=0, sigma=args.sigma, generator=generator)
    validation_seeds = [int(one) for one in str(args.validation_seeds).split(",") if one.strip()]
    parameters = sum(p.numel() for name, p in template.named_parameters() if not name.startswith("critic."))
    run = Run(label=args.label or "genetic", meta={
        "policy": f"genetic algorithm, {parameters/1e6:.2f}M weights a member",
        "task": "movement", "algorithm": "genetic", "population": args.population, "parents": args.parents,
        "sigma": args.sigma, "tasksPerGeneration": args.tasks, "experimentId": args.experiment_id,
        "resumedFrom": str(start_from) if start_from else None, "resumedAt": step,
        "resumedSha256": file_digest(start_from) if start_from else None,
        "goalAbove": world.get("goalAboveShare"), "goalDetour": world.get("goalDetourShare"),
        "goalDig": world.get("goalDigShare"), "goalRadius": full_radius,
        "benchmarkEpisodes": args.benchmark_episodes, "benchmarkSeed": args.benchmark_seed,
        "validationSeeds": validation_seeds, "frameskip": 4, "episodeTicks": args.benchmark_ticks,
    })
    print(f"run {run.id} -> {run.path}", flush=True)
    print(f"evolving {'from ' + str(start_from) if start_from else 'from scratch'}: population "
          f"{args.population}, {args.parents} parents, sigma {args.sigma}, {args.tasks} shared tasks a "
          f"generation, radius {radius:.0f} of {full_radius:.0f}px", flush=True)

    guard = MovementGuard(0)
    if champion_from is not None:
        champion = torch.load(champion_from, map_location="cpu", weights_only=False)
        if champion.get("experimentId") == args.experiment_id and champion.get("validationRoutes"):
            guard.adopt(champion["validationRoutes"])
            (run.path / "best.pt").write_bytes(Path(champion_from).read_bytes())
            # The routes it was examined on come along, so Watch can race it
            # before this process has examined anything.
            source = Path(champion_from).parent
            for suite in [source / "benchmark.json", *sorted(source.glob("validation-*.json"))]:
                if suite.is_file():
                    (run.path / suite.name).write_bytes(suite.read_bytes())
            print(f"champion | kept from {champion_from} at {champion.get('step', 0):,} steps", flush=True)

    def exam(seed, role, network):
        result = run_movement_benchmark(
            network, layout, dict(world, goalRadiusPx=full_radius), levels=world.get("levelFiles") or [],
            generated_maps=world.get("levelPool", 0), episodes=args.benchmark_episodes,
            workers=args.benchmark_workers, envs=args.benchmark_envs, seed=seed,
            episode_ticks=args.benchmark_ticks,
        )
        name = "benchmark.json" if role == "benchmark" else f"{role}-{seed}.json"
        if not (run.path / name).exists():
            (run.path / name).write_text(json.dumps(dict(
                schemaVersion=1, role=role, seed=seed, fingerprint=result.fingerprint, **result.manifest()),
                indent=2) + "\n")
        print(f"{role} | seed {seed} | fixed scenarios {result.reached}/{result.episodes} "
              f"({result.success:.1%}), {result.seconds:.1f}s including failures | "
              f"detours {result.detour_reached}/{len(result.detours)} | "
              f"buried {result.dig_reached}/{len(result.digs)}", flush=True)
        return result

    def check(line, elite_network):
        results = [exam(args.benchmark_seed, "benchmark", elite_network)]
        results.extend(exam(seed, "validation", elite_network) for seed in validation_seeds)
        line.update(results[0].metrics())
        scores = [score_suite(one) for one in results]
        line["validationSuccess"] = sum(s["reached"] for s in scores) / sum(s["episodes"] for s in scores)
        decision = guard.consider(results, seed=generation)
        line["stabilityDecision"] = decision
        if guard.last is not None:
            line.update(guard.last.metrics())
            print(f"selection | against the champion: {guard.last.describe()}", flush=True)
        if decision == "promote":
            line["bestBenchmarkSuccess"] = results[0].success
            line["bestBenchmarkSeconds"] = results[0].seconds
            save(run.path / "best.pt", elite_network, layout, step,
                 benchmarkSuccess=results[0].success, benchmarkSeconds=results[0].seconds,
                 validationSuites=scores, validationRoutes=[one.saved() for one in guard.champion],
                 experimentId=args.experiment_id, generation=generation, algorithm="genetic",
                 goalRadius=radius)
        print(f"stability | {decision}", flush=True)

    try:
        while not args.generations or generation < args.generations:
            generation += 1
            started = time.perf_counter()
            ticks = int(min(args.benchmark_ticks, 60 * (CLOCK_BASE_SECONDS + radius * CLOCK_SECONDS_PER_PX)))
            fitness, success, dig_success, dig_tasks, decisions = score_population(
                population, dict(world, goalRadiusPx=radius), layout,
                seed=args.seed * 1_000_003 + generation, tasks=args.tasks, workers=args.workers,
                per_worker=args.scenarios_per_worker, ticks=ticks,
            )
            step += decisions
            ranked = np.argsort(fitness, kind="stable")
            best = int(ranked[0])
            line = {
                "step": step, "generation": generation, "goalRadiusPx": radius, "clockSeconds": ticks / 60,
                "fitnessBest": float(fitness[best]), "fitnessMedian": float(np.median(fitness)),
                "fitnessEliteKept": float(fitness[0]),
                "fitnessBestSuccess": float(success[best]), "fitnessMeanSuccess": float(success.mean()),
                "fitnessBestDigSuccess": float(dig_success[best]) if dig_tasks else None,
                "digTasks": dig_tasks, "eliteWasKept": int(best == 0),
            }
            print(f"generation {generation:5d} | {step:>13,} steps | radius {radius:6.0f}px clock {ticks/60:4.1f}s | "
                  f"best {fitness[best]:.3f} ({success[best]:.0%} reached"
                  + (f", buried {dig_success[best]:.0%} of {dig_tasks}" if dig_tasks else "")
                  + f") | kept elite {fitness[0]:.3f} | median {np.median(fitness):.3f} "
                  f"| mean reached {success.mean():.0%} | {time.perf_counter() - started:.0f}s", flush=True)
            elite_network = population.member(best)
            # Every --check-every generations, and straight away when this
            # run has no routes yet: Watch races the policy on them.
            if generation % args.check_every == 0 or not (run.path / "benchmark.json").exists():
                check(line, elite_network)
            # The curriculum moves on what the elite managed, among tasks all
            # members shared.
            if success[best] >= args.promote_at:
                radius = min(full_radius, radius * 1.2)
            elif success[best] < args.demote_at:
                radius = max(float(args.goal_radius_start), radius / 1.1)
            line["generationSeconds"] = time.perf_counter() - started
            run.record(**{key: value for key, value in line.items() if value is not None})
            save(run.path / "policy.pt", elite_network, layout, step, experimentId=args.experiment_id,
                 generation=generation, algorithm="genetic", goalRadius=radius)
            parents = [int(i) for i in ranked[: args.parents]]
            population.breed(parents, elite=best, sigma=args.sigma, generator=generator)
    except KeyboardInterrupt:
        run.note("stopped by hand")
        run.close(status="stopped", steps=step)
        raise
    run.close(status="done", steps=step)


def _stop(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    main()
