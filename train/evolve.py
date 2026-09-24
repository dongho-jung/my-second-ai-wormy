"""A genetic algorithm over the movement policy's weights.

No gradients. A population of whole networks is scored on the same tasks, the
best few become parents, and the next generation is the best one kept as it
is plus mutated copies of the parents: every weight moved by Gaussian noise
of size --sigma. This is the "deep neuroevolution" recipe (Such et al., 2017):
truncation selection, elitism, mutation only.

Fitness is what the fixed exam measures: seconds per task with a task given
up costing the whole clock, lower is better. Every member of a generation is
scored on exactly the same tasks. The worlds replay one scenario per
population member (`replicas` in src/env/vec.js), so a lucky draw of short
routes cannot pick a parent. Each generation draws new tasks, so the
population cannot learn one set of routes by heart.

Every --check-every generations the elite takes the same paired exam PPO uses
(train/stability.py) and best.pt moves only when it beats the champion there.
Checkpoints, run.json and metrics.jsonl are the trainer's own, so the monitor,
Watch and `npm run evaluate` read an evolved policy like any other.

    npm run train -- --algorithm genetic --resume artifacts/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import download_seed, experiment_champion, experiment_checkpoint, file_digest
from movement_benchmark import (
    DEFAULT_EPISODES as DEFAULT_BENCHMARK_EPISODES,
    DEFAULT_TICKS,
    fixed_movement_world,
    run_movement_benchmark,
)
from policy import policy_from_shape
from run import Run
from stability import MovementGuard, Routes, compare_routes, score_suite
from workers import WorkerPool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", default="genetic", choices=["genetic"],
                        help="kept so one command line can name the trainer")
    parser.add_argument("--resume", required=True,
                        help="a checkpoint to evolve from, a run's directory, or the runs directory "
                             "together with --experiment-id")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--bootstrap-url", default=None,
                        help="immutable seed URL, used only if this experiment has no checkpoint")
    parser.add_argument("--bootstrap-host", default=None)
    evolution = parser.add_argument_group("evolution")
    evolution.add_argument("--population", type=int, default=24, help="networks scored each generation")
    evolution.add_argument("--parents", type=int, default=6, help="the best this many breed the next generation")
    evolution.add_argument("--sigma", type=float, default=0.002,
                           help="standard deviation of the noise added to every weight of a child")
    evolution.add_argument("--tasks", type=int, default=32,
                           help="tasks every member is scored on each generation (same tasks for all)")
    evolution.add_argument("--generations", type=int, default=0, help="stop after this many; 0 runs on")
    evolution.add_argument("--workers", type=int, default=8)
    evolution.add_argument("--scenarios-per-worker", type=int, default=2,
                           help="distinct tasks a worker plays at once; each is replayed for every member")
    evolution.add_argument("--episode-ticks", type=int, default=DEFAULT_TICKS,
                           help="clock of one task; a task not reached costs all of it")
    exam = parser.add_argument_group("the paired exam that moves best.pt")
    exam.add_argument("--check-every", type=int, default=5, help="generations between exams of the elite")
    exam.add_argument("--benchmark-episodes", type=int, default=DEFAULT_BENCHMARK_EPISODES)
    exam.add_argument("--benchmark-seed", type=int, default=334462)
    exam.add_argument("--validation-seeds", default="1334462")
    exam.add_argument("--test-seed", type=int, default=None,
                      help="scored at the start and on the final best.pt, never used for selection")
    exam.add_argument("--benchmark-workers", type=int, default=8)
    exam.add_argument("--benchmark-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)
    if not 1 <= args.parents <= args.population or args.population < 2:
        parser.error("--population must be at least 2 and --parents between 1 and --population")
    if args.sigma <= 0 or args.tasks < 1 or args.check_every < 1:
        parser.error("--sigma, --tasks and --check-every must be positive")
    return args


def locate(args):
    """The checkpoint to start from, and the champion a restarted experiment was keeping."""
    resume = Path(args.resume)
    if args.experiment_id and resume.is_dir():
        found = experiment_checkpoint(resume, args.experiment_id)
        champion = experiment_champion(resume, args.experiment_id)
        if found is None and args.bootstrap_url:
            found = download_seed(args.bootstrap_url, resume, args.experiment_id, args.bootstrap_host)
        if found is not None:
            return found, champion
    if resume.is_file():
        return resume, None
    for name in ("best.pt", "policy.pt"):
        if (resume / name).is_file():
            return resume / name, None
    raise SystemExit(f"nothing to evolve from at {resume}")


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


class Genome:
    """The weights evolution moves: every parameter, not the input normaliser's statistics."""

    def __init__(self, policy):
        self.names = [name for name, _ in policy.named_parameters()]

    def mutate(self, parent, child, sigma, generator):
        with torch.no_grad():
            for (_, mine), (_, theirs) in zip(child.named_parameters(), parent.named_parameters()):
                mine.copy_(theirs + sigma * torch.randn(theirs.shape, generator=generator, dtype=theirs.dtype))
            for mine, theirs in zip(child.buffers(), parent.buffers()):
                mine.copy_(theirs)


def score_population(members, world, layout, *, seed, tasks, workers, per_worker, episode_ticks, device):
    """Every member on the same tasks: seconds per task (failures at the clock) and tasks reached.

    World i of a worker plays scenario floor(i / P) and belongs to member i % P,
    so the P replicas of a scenario are one per member. Scenario (worker w, slot
    s, episode k) is the same for all of them whenever it finishes; results are
    kept by that key, so a member that arrives early is not scored on different
    tasks from one that does not.
    """
    population = len(members)
    envs = per_worker * population
    config = fixed_movement_world(
        world, layout, levels=world.get("levelFiles") or [], generated_maps=world.get("levelPool", 0),
        envs=envs, seed=seed, episode_ticks=episode_ticks, end_on_arrival=True,
    )
    config["replicas"] = population
    pool = WorkerPool(workers, config)
    shape = pool.layout
    if shape.vector_size != layout["vectorSize"] or shape.head_sizes != list(layout["headSizes"]):
        pool.close()
        raise RuntimeError("the evolution worlds do not match the policy's observation or actions")
    fields = {name: index for index, name in enumerate(shape.stat_fields)}
    done_first = shape.done_codes["first"]
    closing = (shape.done_codes["last"], shape.done_codes.get("terminal", shape.done_codes["last"]))
    worlds = pool.envs
    scenarios = worlds // population
    rounds = -(-tasks // scenarios)
    lanes = [torch.arange(member, worlds, population) for member in range(population)]
    memory = torch.zeros(worlds, members[0].memory_width)
    finished = np.zeros(worlds, dtype=np.int64)
    seconds = np.full((population, scenarios * rounds), np.nan)
    reached = np.zeros((population, scenarios * rounds))
    use_patch = shape.patch_cells > 0 and members[0].use_patch
    use_map = shape.map_cells > 0 and members[0].use_map
    vectors, patches, maps, _, _, _, _ = pool.observations()
    done = np.zeros(worlds)
    restarts = np.zeros(worlds)
    decisions = 0
    try:
        while np.isnan(seconds).any():
            v = torch.as_tensor(vectors)
            p = torch.as_tensor(patches) if use_patch else None
            m = torch.as_tensor(maps) if use_map else None
            restart = torch.as_tensor(((done == done_first) | (restarts > 0)).astype(np.float32))
            heads = torch.zeros(worlds, len(shape.head_sizes), dtype=torch.long)
            with torch.no_grad():
                for member, lane in zip(members, lanes):
                    logits, _, kept = member(
                        v[lane], p[lane] if p is not None else None, m[lane] if m is not None else None,
                        carried=memory[lane], restart=restart[lane],
                    )
                    memory[lane] = kept
                    heads[lane] = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
            pool.step(heads.to(torch.uint8).numpy())
            vectors, patches, maps, _, env_done, restarts, stats = pool.observations()
            done = env_done
            decisions += worlds
            for world in np.nonzero(np.isin(env_done, closing))[0]:
                episode = int(finished[world])
                finished[world] += 1
                if episode >= rounds:
                    continue
                member = int(world % population)
                worker, slot = divmod(int(world), envs)
                task = episode * scenarios + worker * per_worker + slot // population
                row = stats[world]
                hit = float(row[fields["goalsReached"]]) > 0.5
                reached[member, task] = hit
                seconds[member, task] = float(row[fields["goalSeconds"]]) if hit else episode_ticks / 60
    finally:
        pool.close()
    take = slice(0, tasks)
    return seconds[:, take].mean(axis=1), reached[:, take].mean(axis=1), decisions


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    start_from, champion_from = locate(args)
    carried = torch.load(start_from, map_location="cpu", weights_only=False)
    layout = carried["layout"]
    world = dict(layout.get("world") or {})
    if not world:
        raise SystemExit(f"{start_from} does not say which world it learned in")
    elite = policy_from_shape(layout)
    elite.load_state_dict(carried["policy"])
    elite.eval()
    step = int(carried.get("step", 0))
    genome = Genome(elite)
    generator = torch.Generator().manual_seed(args.seed)
    validation_seeds = [int(one) for one in str(args.validation_seeds).split(",") if one.strip()]
    run = Run(label=args.label or "genetic", meta={
        "policy": f"genetic algorithm, {sum(p.numel() for p in elite.parameters())/1e6:.2f}M weights",
        "task": "movement", "algorithm": "genetic", "population": args.population, "parents": args.parents,
        "sigma": args.sigma, "tasksPerGeneration": args.tasks, "experimentId": args.experiment_id,
        "resumedFrom": str(start_from), "resumedAt": step, "resumedSha256": file_digest(start_from),
        "benchmarkEpisodes": args.benchmark_episodes, "benchmarkSeed": args.benchmark_seed,
        "validationSeeds": validation_seeds, "testSeed": args.test_seed,
        "episodeTicks": layout.get("episodeTicks"), "frameskip": layout.get("frameskip"),
    })
    print(f"run {run.id} -> {run.path}", flush=True)
    print(f"evolving from {start_from} at {step:,} steps: population {args.population}, "
          f"{args.parents} parents, sigma {args.sigma}, {args.tasks} shared tasks a generation", flush=True)

    guard = MovementGuard(0)
    if champion_from is not None:
        champion = torch.load(champion_from, map_location="cpu", weights_only=False)
        if champion.get("experimentId") == args.experiment_id and champion.get("validationRoutes"):
            guard.adopt(champion["validationRoutes"])
            (run.path / "best.pt").write_bytes(Path(champion_from).read_bytes())
            print(f"champion | kept from {champion_from} at {champion.get('step', 0):,} steps", flush=True)

    def exam(seed, role, network):
        result = run_movement_benchmark(
            network, layout, world, levels=world.get("levelFiles") or [],
            generated_maps=world.get("levelPool", 0), episodes=args.benchmark_episodes,
            workers=args.benchmark_workers, envs=args.benchmark_envs, seed=seed,
            episode_ticks=args.episode_ticks,
        )
        name = "benchmark.json" if role == "benchmark" else f"{role}-{seed}.json"
        if not (run.path / name).exists():
            (run.path / name).write_text(json.dumps(dict(
                schemaVersion=1, role=role, seed=seed, fingerprint=result.fingerprint, **result.manifest()),
                indent=2) + "\n")
        print(f"{role} | seed {seed} | fixed scenarios {result.reached}/{result.episodes} "
              f"({result.success:.1%}), {result.seconds:.1f}s including failures | "
              f"detours {result.detour_reached}/{len(result.detours)} ({result.detour_success:.1%})", flush=True)
        return result

    def check(line, generation):
        results = [exam(args.benchmark_seed, "benchmark", elite)]
        results.extend(exam(seed, "validation", elite) for seed in validation_seeds)
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
            save(run.path / "best.pt", elite, layout, step,
                 benchmarkSuccess=results[0].success, benchmarkSeconds=results[0].seconds,
                 validationSuites=scores, validationRoutes=[one.saved() for one in guard.champion],
                 experimentId=args.experiment_id, generation=generation, algorithm="genetic")
        print(f"stability | {decision}", flush=True)

    first_test = None
    if args.test_seed is not None:
        first_test = Routes.of(exam(args.test_seed, "test", elite))
    initial = {"step": step, "generation": 0}
    check(initial, 0)
    run.record(**initial)

    members = [copy.deepcopy(elite) for _ in range(args.population)]
    parents = [copy.deepcopy(elite)]
    generation = 0
    try:
        while not args.generations or generation < args.generations:
            generation += 1
            started = time.perf_counter()
            # The elite goes in untouched and is scored again with everyone else:
            # a lucky score from last generation does not carry over.
            members[0].load_state_dict(elite.state_dict())
            for child in members[1:]:
                genome.mutate(random.choice(parents), child, args.sigma, generator)
            fitness, success, decisions = score_population(
                members, world, layout, seed=args.seed * 1_000_003 + generation, tasks=args.tasks,
                workers=args.workers, per_worker=args.scenarios_per_worker,
                episode_ticks=args.episode_ticks, device=args.device,
            )
            step += decisions
            order = np.argsort(fitness, kind="stable")
            ranked = [members[i] for i in order]
            elite.load_state_dict(ranked[0].state_dict())
            parents = [copy.deepcopy(one) for one in ranked[: args.parents]]
            line = {
                "step": step, "generation": generation,
                "fitnessBestSeconds": float(fitness[order[0]]),
                "fitnessMedianSeconds": float(np.median(fitness)),
                "fitnessEliteKeptSeconds": float(fitness[0]),
                "fitnessBestSuccess": float(success[order[0]]),
                "fitnessMeanSuccess": float(success.mean()),
                "eliteWasKept": int(order[0] == 0),
                "generationSeconds": time.perf_counter() - started,
            }
            print(f"generation {generation:4d} | {step:>12,} steps | best {line['fitnessBestSeconds']:5.2f}s "
                  f"{line['fitnessBestSuccess']:.0%} | kept elite {line['fitnessEliteKeptSeconds']:5.2f}s | "
                  f"median {line['fitnessMedianSeconds']:5.2f}s | mean success {line['fitnessMeanSuccess']:.0%} | "
                  f"{'elite kept' if order[0] == 0 else 'new elite'} | {line['generationSeconds']:.0f}s", flush=True)
            if generation % args.check_every == 0:
                check(line, generation)
            run.record(**line)
            save(run.path / "policy.pt", elite, layout, step, experimentId=args.experiment_id,
                 generation=generation, algorithm="genetic")
    except KeyboardInterrupt:
        run.note("stopped by hand")
        run.close(status="stopped", steps=step)
        raise
    if first_test is not None and (run.path / "best.pt").exists():
        chosen = policy_from_shape(layout)
        chosen.load_state_dict(torch.load(run.path / "best.pt", weights_only=False)["policy"])
        against = compare_routes([Routes.of(exam(args.test_seed, "test", chosen))], [first_test])
        print(f"test | selected against the start: {against.describe()}", flush=True)
        run.record(step=step, **against.metrics("testStart"))
    run.close(status="done", steps=step)


def _stop(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    main()
