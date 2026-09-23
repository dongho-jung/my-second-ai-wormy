"""Two policies in the same matches, and which one comes out ahead.

Nothing on the training page can compare two runs. Every figure there is a
policy measured against itself, or against its own recent past, and a run that
improves slowly and one that improves quickly can show the same "beating its
past self" number. So this seats one checkpoint against another — or against a
fixed baseline — in the same free-for-all, on the same maps, with the sides
swapped every other match, and reports the difference with a confidence
interval.

    npm run evaluate -- --left artifacts/runs/<a>/best.pt --right artifacts/runs/<b>/best.pt
    npm run evaluate -- --left artifacts/runs/<a> --right random
    npm run evaluate -- --left artifacts/runs/<a> --right still --episodes 96

A run directory stands for its best.pt, and `latest` for the newest run's.
`random` presses keys at random and
`still` presses nothing: two bars that never move, so a policy can be measured
against the same thing at the start of a run and at the end of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy import policy_from_shape
from ppo import stock_levels
from run import DEFAULT_RUNS
from watch import newest_checkpoint
from workers import REPO, WorkerPool

BASELINES = ("random", "still")

# What is compared, as the all-worm mean the worker reports and the front-minus
# back difference beside it. Deaths and self damage are in the list so that
# "kills more" is not mistaken for "plays better" when it also dies more.
COMBAT_METRICS = [
    ("kills", "kills", "killsVsPast"),
    ("deaths", "deaths", "deathsVsPast"),
    ("damage dealt", "damageDealt", "damageVsPast"),
    ("damage to itself", "selfDamage", "selfDamageVsPast"),
    ("deaths, own doing", "suicides", "suicidesVsPast"),
]

# A destination is private to one worm, but the paired seating still matters:
# the policies dig the same terrain and can block or move one another. The
# front-minus-back fields let the evaluator recover each side after the swap,
# exactly as for combat. Reached is primary because a sixty-second episode
# makes it goals/minute; the other three explain whether a win came from faster,
# straighter travel or merely easier goals.
MOVEMENT_METRICS = [
    ("destinations reached", "goalsReached", "goalsVsPast"),
    ("seconds per destination", "goalSeconds", "goalSecondsVsPast"),
    ("direct speed, px/s", "goalSpeed", "goalSpeedVsPast"),
    ("path efficiency", "goalPathEfficiency", "goalEfficiencyVsPast"),
    ("destinations missed", "goalsMissed", "goalsMissedVsPast"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", default=None,
                        help="a .pt, a run directory, `latest` for the newest run, or random / still")
    parser.add_argument("--right", default=None, help="the other side, the same way")
    parser.add_argument("--history", default=None,
                        help="a run directory: its best.pt against every policy-<steps>.pt the run "
                             "kept (--keep-every), oldest first, so the run's progress is measured "
                             "against its own earlier selves rather than read off the reward")
    parser.add_argument("--episodes", type=int, default=48, help="finished matches to count")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--envs", type=int, default=6, help="matches per worker, played at once")
    parser.add_argument("--agents", type=int, default=None,
                        help="worms in a match; the number the left side trained with by default")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--greedy", action="store_true",
                        help="each side takes its most likely key instead of sampling. Training "
                             "samples, so sampling is the fairer default")
    parser.add_argument("--levels-dir", default=str(REPO / "artifacts" / "maps" / "dsds-cs"),
                        help="the maps to play on. The checkpoint's own list is not used: its paths "
                             "belong to the machine it trained on")
    parser.add_argument("--stock-levels", type=int, default=64, help="how many of them; 0 for none")
    parser.add_argument("--maps", type=int, default=0, help="generated dirt levels to mix in")
    parser.add_argument("--episode-ticks", type=int, default=None)
    parser.add_argument("--task", default="auto", choices=["auto", "fight", "movement"],
                        help="metrics to compare. auto reads whether the checkpoint's world has goals")
    parser.add_argument("--goal-radius", type=float, default=None,
                        help="fixed movement-evaluation radius. By default the far end of the "
                             "checkpoint's range; no curriculum runs during an evaluation")
    parser.add_argument("--goal-patience", type=int, default=None,
                        help="fixed movement-evaluation deadline in decisions. By default the "
                             "checkpoint's original maximum; no deadline curriculum runs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--json", default=None, help="also write the result here")
    return parser.parse_args(argv)


def checkpoint_path(spec: str) -> Path:
    """A .pt as given, a run directory's best.pt or policy.pt, or `latest`."""
    if spec == "latest":
        return newest_checkpoint(DEFAULT_RUNS)
    path = Path(spec)
    if path.is_dir():
        for name in ("best.pt", "policy.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"{path} has no best.pt or policy.pt")
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


class Side:
    """One side of the match: a checkpoint with its memory, or a baseline."""

    def __init__(self, spec: str, device, greedy: bool, head_sizes=None):
        self.name = spec
        self.greedy = greedy
        self.shape = None
        self.policy = None
        self.memory = None
        self.head_sizes = head_sizes
        # Pixels per patch cell this side trained at, and whether it takes the
        # match's second cut of the ground rather than the first.
        self.patch_scale = None
        self.patch2 = False
        if spec in BASELINES:
            return
        path = checkpoint_path(spec)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        shape = checkpoint["layout"]
        self.name = f"{path.parent.name}/{path.name}"
        self.shape = shape
        self.steps = int(checkpoint.get("step", 0))
        self.policy = policy_from_shape(shape).to(device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()
        self.head_sizes = list(shape["headSizes"])
        world = shape.get("world") or {}
        rows = (shape.get("patchShape") or [6, 121, 213])[-2]
        self.patch_scale = int(world.get("patchScale") or round(242 / rows))

    @property
    def baseline(self) -> bool:
        return self.policy is None

    def start(self, pools: int, slots: int, device):
        if self.policy is not None:
            self.memory = [
                torch.zeros(slots, self.policy.memory_width, device=device) for _ in range(pools)
            ]

    def act(self, pool, index, vectors, patches, maps, restart):
        """Heads for the slots in `index` of one pool, remembering only for those slots."""
        count = len(index)
        if self.name == "still":
            return torch.zeros(count, len(self.head_sizes), dtype=torch.long)
        if self.name == "random":
            return torch.stack(
                [torch.randint(0, size, (count,)) for size in self.head_sizes], dim=1
            )
        with torch.no_grad():
            carried = self.memory[pool][index]
            take = lambda block: block[index] if block is not None else None
            if self.greedy:
                logits, _, kept = self.policy(
                    take(vectors), take(patches), take(maps), carried, restart[index]
                )
                heads = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
            else:
                heads, _, _, _, kept = self.policy.act(
                    take(vectors), take(patches), take(maps),
                    want_entropy=False, carried=carried, restart=restart[index],
                )
            self.memory[pool][index] = kept
        return heads


def bootstrap(values, resamples=5000, seed=0):
    """A 95% interval on the mean, by resampling the episodes."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    if args.history:
        return history(args)
    if not args.left or not args.right:
        raise SystemExit("give --left and --right, or --history RUN_DIR")
    return compare(args, args.left, args.right)


def history(args):
    """A run against its kept checkpoints, oldest first."""
    run = Path(args.history)
    kept = sorted(
        run.glob("policy-*.pt"),
        key=lambda path: int(path.stem.split("-", 1)[1]),
    )
    if not kept:
        raise SystemExit(f"{run} kept no policy-<steps>.pt: train with --keep-every to have some")
    final = checkpoint_path(str(run))
    print(f"{final.name} of {run.name} against the {len(kept)} checkpoints the run kept", flush=True)
    results = []
    for path in kept:
        print(f"\n--- against {path.name} ---", flush=True)
        results.append((path, compare(args, str(final), str(path))))
    print()
    primary_label = results[0][1]["primary"] if results else "primary score"
    print(f"{'earlier self':>22s}{(primary_label + ', later - earlier'):>34s}{'95% interval':>20s}{'pairs won':>12s}")
    for path, result in results:
        primary = result["metrics"][result["primary"]]
        low, high = primary["interval"]
        pairs = result["pairs"]
        print(
            f"{path.stem.split('-', 1)[1]:>22s}{primary['difference']:+34.2f}"
            f"{'[' + f'{low:+.2f}, {high:+.2f}' + ']':>20s}"
            f"{pairs['left_ahead']:>6d}/{result['episodes']:<5d}"
        )
    # Kept beside the checkpoints it is about, so the measurement is not lost
    # in a terminal: each entry is one comparison, oldest self first.
    out = Path(args.json) if args.json else run / "history.json"
    out.write_text(json.dumps(
        {"final": final.name, "primary": results[0][1]["primary"] if results else None, "against": [
            {"checkpoint": path.name, "steps": int(path.stem.split("-", 1)[1]), **result}
            for path, result in results
        ]},
        indent=2,
    ) + "\n")
    print(f"written to {out}")
    return results


def compare(args, left_spec, right_spec):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    left = Side(left_spec, device, args.greedy)
    right = Side(right_spec, device, args.greedy, head_sizes=left.head_sizes)
    if left.baseline and right.baseline:
        raise SystemExit("at least one side has to be a checkpoint: the match is built from its world")
    if left.baseline:
        left.head_sizes = right.head_sizes
    anchor = right if left.baseline else left
    shape = anchor.shape
    checkpoint_world = shape.get("world") or {}
    movement = (
        args.task == "movement"
        or (args.task == "auto" and bool(checkpoint_world.get("goals")))
    )
    metrics = MOVEMENT_METRICS if movement else COMBAT_METRICS
    for side in (left, right):
        if side.baseline or side is anchor:
            continue
        for key in ("vectorSize", "headSizes", "usePatch", "useMap", "mapSide"):
            if side.shape.get(key) != shape.get(key):
                raise SystemExit(
                    f"{left.name} and {right.name} cannot sit in one match: {key} is "
                    f"{shape.get(key)} on one side and {side.shape.get(key)} on the other. "
                    "They were trained on different observations, and a match encodes one"
                )
    # Two policies that look at the ground through different patch scales can
    # still share a match: the world cuts the same ground twice and each side
    # is shown the cut it learned on.
    second_scale = None
    if not left.baseline and not right.baseline and left.patch_scale != right.patch_scale:
        second_scale = right.patch_scale
        right.patch2 = True

    trained_agents = int(shape.get("agents", 3))
    agents = args.agents or trained_agents
    if agents < 2:
        raise SystemExit("a comparison needs at least two worms in a match")
    # The back seats of every match are one side, the front seats the other.
    # Two pools of worlds play the same seeds — the same maps, spawns, loadouts
    # and delays, in the same order — with the sides swapped between them, so
    # every map is played once each way and the pair is one sample: a seat
    # effect cancels, and the map's own swing (a cramped map is a bloodbath
    # for everybody) drops out of the difference instead of widening it.
    back = agents // 2
    front = agents - back

    config = {
        **checkpoint_world,
        "agents": agents,
        "observationFoes": trained_agents - 1,
        "envs": args.envs,
        "seed": args.seed,
        # Whole episodes, all counted; the trainer's staggered start is for
        # spreading statistics over time, which is the opposite of wanted here.
        "stagger": False,
        # No ladder: what is counted is what the game scores, and the reward
        # terms are not read anyway.
        "shapingFullAt": 0,
        "decisionsDone": 0,
        "opponents": back,
        "levelPool": args.maps,
        "levelFiles": stock_levels(args),
    }
    if second_scale:
        config["patchScale2"] = second_scale
    if args.episode_ticks:
        config["episodeTicks"] = args.episode_ticks
    if movement:
        radius = config.get("goalRadiusPx")
        if args.goal_radius is not None:
            radius = args.goal_radius
        elif isinstance(radius, list):
            radius = max(radius)
        config["goalRadiusPx"] = radius
        config["goalRadiusMode"] = "steps"
        config.pop("goalCurriculum", None)
        config.pop("goalRadiusStart", None)
        if args.goal_patience is not None:
            config["goalPatience"] = args.goal_patience
        config.pop("goalPatienceMin", None)
        config.pop("goalPatienceStart", None)
        config.pop("goalDeadlineCurriculum", None)

    pools = [WorkerPool(args.workers, config) for _ in range(2)]
    layout = pools[0].layout
    if layout.vector_size != shape["vectorSize"] or layout.head_sizes != list(shape["headSizes"]):
        for pool in pools:
            pool.close()
        raise SystemExit(
            f"the checkpoint wants a vector of {shape['vectorSize']} with heads {shape['headSizes']} "
            f"and the match gives {layout.vector_size} with heads {layout.head_sizes}"
        )
    slots = pools[0].slots
    per_match = layout.agents
    matches = pools[0].envs
    use_patch = bool(shape.get("usePatch", True)) and layout.patch_cells > 0
    use_map = bool(shape.get("useMap", False)) and layout.map_cells > 0
    if second_scale and layout.patch2_cells != right.shape["patchShape"][-2] * right.shape["patchShape"][-1]:
        for pool in pools:
            pool.close()
        raise SystemExit(
            f"the match cuts the second patch into {layout.patch2_cells} cells and "
            f"{right.name} wants {right.shape['patchShape']}"
        )
    at = {name: index for index, name in enumerate(layout.stat_fields)}
    DONE_FIRST = layout.done_codes["first"]
    DONE_LAST = layout.done_codes["last"]

    # Pool 0 seats the left side in front, pool 1 seats it behind.
    front_slots = [slot for match in range(matches) for slot in range(match * per_match, match * per_match + front)]
    back_slots = [slot for match in range(matches) for slot in range(match * per_match + front, (match + 1) * per_match)]
    seating = [
        {"left": torch.tensor(front_slots), "right": torch.tensor(back_slots), "left_front": True},
        {"left": torch.tensor(back_slots), "right": torch.tensor(front_slots), "left_front": False},
    ]
    for side in (left, right):
        side.start(2, slots, device)

    print(
        f"{left.name} (left) against {right.name} (right): {agents} worms a match, "
        f"{front} in front and {back} behind, every map played once each way, "
        f"2 x {matches} matches at once on {layout.maps} maps"
        + (f", the ground cut at {left.patch_scale} px a cell for the left and "
           f"{right.patch_scale} for the right" if second_scale else "")
        + (", greedy" if args.greedy else ""),
        flush=True,
    )

    class Stream:
        """One pool's latest observation, as tensors."""

        def __init__(self, pool):
            self.pool = pool
            vectors, patches, maps, _, _, _, _ = pool.observations()
            self.v = torch.as_tensor(vectors, device=device)
            self.p = torch.as_tensor(patches, device=device) if use_patch else None
            self.p2 = torch.as_tensor(pool.patches2, device=device) if second_scale else None
            self.m = torch.as_tensor(maps, device=device) if use_map else None
            self.done = torch.zeros(slots, device=device)
            self.reset = torch.zeros(slots, device=device)
            self.heads = torch.zeros(slots, len(layout.head_sizes), dtype=torch.long)

        def step(self, index):
            restart = ((self.done == DONE_FIRST) | (self.reset > 0)).float()
            seats = seating[index]
            cut = lambda side: self.p2 if side.patch2 else self.p
            self.heads[seats["left"]] = left.act(index, seats["left"], self.v, cut(left), self.m, restart)
            self.heads[seats["right"]] = right.act(index, seats["right"], self.v, cut(right), self.m, restart)
            self.pool.step(self.heads.to(torch.uint8).numpy())
            vectors, patches, maps, _, env_done, restarts, stats = self.pool.observations()
            self.v = torch.as_tensor(vectors, device=device)
            if use_patch:
                self.p = torch.as_tensor(patches, device=device)
            if second_scale:
                self.p2 = torch.as_tensor(self.pool.patches2, device=device)
            if use_map:
                self.m = torch.as_tensor(maps, device=device)
            self.done = torch.as_tensor(np.repeat(env_done, per_match).astype(np.float32), device=device)
            self.reset = torch.as_tensor(restarts.astype(np.float32), device=device)
            return env_done, stats

    streams = [Stream(pool) for pool in pools]

    def sides_of(row, left_front):
        """A stats row's all-worm mean and front-minus-back difference, as the
        left side's mean and the right side's."""
        out = {}
        n_left = front if left_front else back
        n_right = per_match - n_left
        sign = 1.0 if left_front else -1.0
        for label, mean_field, versus_field in metrics:
            mean = float(row[at[mean_field]])
            diff = sign * float(row[at[versus_field]])
            out[label] = (mean + n_right * diff / per_match, mean - n_left * diff / per_match)
        return out

    # Per finished pair of episodes, per metric: the left side's mean and the
    # right's, each averaged over the two seatings.
    rows = {label: {"left": [], "right": []} for label, _, _ in metrics}
    finished = 0
    started = time.perf_counter()
    steps = 0
    try:
        while finished < args.episodes:
            outcomes = [stream.step(index) for index, stream in enumerate(streams)]
            steps += 1
            closed = [np.nonzero(done == DONE_LAST)[0] for done, _ in outcomes]
            if not np.array_equal(closed[0], closed[1]):
                raise RuntimeError(
                    "the two pools have drifted apart: an episode closed in one and not the "
                    f"other at step {steps} ({closed[0].tolist()} against {closed[1].tolist()})"
                )
            for match in closed[0]:
                if finished >= args.episodes:
                    break
                once = sides_of(outcomes[0][1][match], seating[0]["left_front"])
                twice = sides_of(outcomes[1][1][match], seating[1]["left_front"])
                for label, _, _ in metrics:
                    rows[label]["left"].append((once[label][0] + twice[label][0]) / 2)
                    rows[label]["right"].append((once[label][1] + twice[label][1]) / 2)
                finished += 1
                if finished % 8 == 0 or finished == args.episodes:
                    primary = metrics[0][0]
                    score_left = np.mean(rows[primary]["left"])
                    score_right = np.mean(rows[primary]["right"])
                    print(
                        f"{finished:4d}/{args.episodes} paired episodes | {primary} "
                        f"left {score_left:5.2f} right {score_right:5.2f} | "
                        f"{steps * slots * 2 / (time.perf_counter() - started):,.0f} steps/s",
                        flush=True,
                    )
    finally:
        for pool in pools:
            pool.close()

    result = {
        "left": left.name,
        "right": right.name,
        "episodes": finished,
        "agents": agents,
        "front": front,
        "back": back,
        "maps": layout.maps,
        "paired": True,
        "greedy": args.greedy,
        "seed": args.seed,
        "task": "movement" if movement else "fight",
        "primary": metrics[0][0],
        "metrics": {},
    }
    print()
    print(f"{'':18s}{'left':>9s}{'right':>9s}{'left - right':>15s}{'95% interval':>22s}")
    for label, _, _ in metrics:
        lefts = np.asarray(rows[label]["left"])
        rights = np.asarray(rows[label]["right"])
        diffs = lefts - rights
        low, high = bootstrap(diffs, seed=args.seed)
        result["metrics"][label] = {
            "left": float(lefts.mean()),
            "right": float(rights.mean()),
            "difference": float(diffs.mean()),
            "interval": [low, high],
        }
        print(
            f"{label:18s}{lefts.mean():9.2f}{rights.mean():9.2f}{diffs.mean():+15.2f}"
            f"{'[' + f'{low:+.2f}, {high:+.2f}' + ']':>22s}"
        )
    # Pair by pair, who won the task's primary score.
    primary_label = metrics[0][0]
    lefts = np.asarray(rows[primary_label]["left"])
    rights = np.asarray(rows[primary_label]["right"])
    ahead = int((lefts > rights).sum())
    behind = int((lefts < rights).sum())
    level = int(len(lefts) - ahead - behind)
    result["pairs"] = {"left_ahead": ahead, "right_ahead": behind, "level": level}
    primary = result["metrics"][primary_label]
    low, high = primary["interval"]
    if low > 0:
        verdict = f"{left.name} is ahead on {primary_label}"
    elif high < 0:
        verdict = f"{right.name} is ahead on {primary_label}"
    else:
        verdict = f"no difference on {primary_label} that these paired episodes can tell apart"
    result["verdict"] = verdict
    print()
    print(f"left scored higher in {ahead} of {len(lefts)} paired episodes, lower in {behind}, the same in {level}")
    print(verdict, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"written to {args.json}")
    return result


if __name__ == "__main__":
    main()
