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

A run directory stands for its best.pt. `random` presses keys at random and
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
from workers import REPO, WorkerPool

BASELINES = ("random", "still")

# What is compared, as the all-worm mean the worker reports and the front-minus
# back difference beside it. Deaths and self damage are in the list so that
# "kills more" is not mistaken for "plays better" when it also dies more.
METRICS = [
    ("kills", "kills", "killsVsPast"),
    ("deaths", "deaths", "deathsVsPast"),
    ("damage dealt", "damageDealt", "damageVsPast"),
    ("damage to itself", "selfDamage", "selfDamageVsPast"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", required=True, help="a .pt, a run directory, or random / still")
    parser.add_argument("--right", required=True, help="the other side, the same way")
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
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--json", default=None, help="also write the result here")
    return parser.parse_args(argv)


def checkpoint_path(spec: str) -> Path:
    """A .pt as given, or a run directory's best.pt, or its policy.pt."""
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

    @property
    def baseline(self) -> bool:
        return self.policy is None

    def start(self, slots: int, device):
        if self.policy is not None:
            self.memory = torch.zeros(slots, self.policy.memory_width, device=device)

    def act(self, index, vectors, patches, maps, restart):
        """Heads for the slots in `index`, remembering only for those slots."""
        count = len(index)
        if self.name == "still":
            return torch.zeros(count, len(self.head_sizes), dtype=torch.long)
        if self.name == "random":
            return torch.stack(
                [torch.randint(0, size, (count,)) for size in self.head_sizes], dim=1
            )
        with torch.no_grad():
            carried = self.memory[index]
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
            self.memory[index] = kept
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)

    left = Side(args.left, device, args.greedy)
    right = Side(args.right, device, args.greedy, head_sizes=left.head_sizes)
    if left.baseline and right.baseline:
        raise SystemExit("at least one side has to be a checkpoint: the match is built from its world")
    if left.baseline:
        left.head_sizes = right.head_sizes
    anchor = right if left.baseline else left
    shape = anchor.shape
    for side in (left, right):
        if side.baseline or side is anchor:
            continue
        for key in ("vectorSize", "headSizes", "patchShape", "usePatch", "useMap", "mapSide", "convPadding"):
            if side.shape.get(key) != shape.get(key):
                raise SystemExit(
                    f"{left.name} and {right.name} cannot sit in one match: {key} is "
                    f"{shape.get(key)} on one side and {side.shape.get(key)} on the other. "
                    "They were trained on different observations, and a match encodes one"
                    + (" — two patch scales would need an observation per side, which does "
                       "not exist yet" if key == "patchShape" else "")
                )

    trained_agents = int(shape.get("agents", 3))
    agents = args.agents or trained_agents
    if agents < 2:
        raise SystemExit("a comparison needs at least two worms in a match")
    # The back seats of every match are one side, the front seats the other,
    # and which side that is alternates from match to match, so a seat effect —
    # spawn order, worm colour — cancels instead of accumulating.
    back = agents // 2
    front = agents - back

    config = {
        **(shape.get("world") or {}),
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
        "opponents": back,
        "levelPool": args.maps,
        "levelFiles": stock_levels(args),
    }
    if args.episode_ticks:
        config["episodeTicks"] = args.episode_ticks

    pool = WorkerPool(args.workers, config)
    layout = pool.layout
    if layout.vector_size != shape["vectorSize"] or layout.head_sizes != list(shape["headSizes"]):
        pool.close()
        raise SystemExit(
            f"the checkpoint wants a vector of {shape['vectorSize']} with heads {shape['headSizes']} "
            f"and the match gives {layout.vector_size} with heads {layout.head_sizes}"
        )
    slots = pool.slots
    per_match = layout.agents
    matches = pool.envs
    use_patch = bool(shape.get("usePatch", True)) and layout.patch_cells > 0
    use_map = bool(shape.get("useMap", False)) and layout.map_cells > 0
    at = {name: index for index, name in enumerate(layout.stat_fields)}
    DONE_FIRST = layout.done_codes["first"]
    DONE_LAST = layout.done_codes["last"]

    # Who sits where. Even matches: left in front. Odd matches: right in front.
    left_slots, right_slots, left_in_front = [], [], []
    for match in range(matches):
        base = match * per_match
        front_slots = list(range(base, base + front))
        back_slots = list(range(base + front, base + per_match))
        if match % 2 == 0:
            left_slots += front_slots
            right_slots += back_slots
            left_in_front.append(True)
        else:
            left_slots += back_slots
            right_slots += front_slots
            left_in_front.append(False)
    left_index = torch.tensor(left_slots, dtype=torch.long)
    right_index = torch.tensor(right_slots, dtype=torch.long)
    left.start(slots, device)
    right.start(slots, device)

    print(
        f"{left.name} (left) against {right.name} (right): {agents} worms a match, "
        f"{front} in front and {back} behind, sides swapped every other match, "
        f"{matches} matches at once on {layout.maps} maps"
        + (", greedy" if args.greedy else ""),
        flush=True,
    )

    vectors, patches, maps, _, _, _, _ = pool.observations()
    next_v = torch.as_tensor(vectors, device=device)
    next_p = torch.as_tensor(patches, device=device) if use_patch else None
    next_m = torch.as_tensor(maps, device=device) if use_map else None
    next_done = torch.zeros(slots, device=device)
    next_reset = torch.zeros(slots, device=device)

    # Per finished episode, per metric: the left side's mean and the right's.
    rows = {label: {"left": [], "right": []} for label, _, _ in METRICS}
    finished = 0
    started = time.perf_counter()
    steps = 0
    heads = torch.zeros(slots, len(layout.head_sizes), dtype=torch.long)
    try:
        while finished < args.episodes:
            restart = ((next_done == DONE_FIRST) | (next_reset > 0)).float()
            heads[left_index] = left.act(left_index, next_v, next_p, next_m, restart)
            heads[right_index] = right.act(right_index, next_v, next_p, next_m, restart)
            pool.step(heads.to(torch.uint8).numpy())
            vectors, patches, maps, _, env_done, restarts, stats = pool.observations()
            next_v = torch.as_tensor(vectors, device=device)
            if use_patch:
                next_p = torch.as_tensor(patches, device=device)
            if use_map:
                next_m = torch.as_tensor(maps, device=device)
            next_done = torch.as_tensor(np.repeat(env_done, per_match).astype(np.float32), device=device)
            next_reset = torch.as_tensor(restarts.astype(np.float32), device=device)
            steps += 1
            for match in np.nonzero(env_done == DONE_LAST)[0]:
                if finished >= args.episodes:
                    break
                row = stats[match]
                n_left = front if left_in_front[match] else back
                n_right = per_match - n_left
                sign = 1.0 if left_in_front[match] else -1.0
                for label, mean_field, versus_field in METRICS:
                    mean = float(row[at[mean_field]])
                    # front minus back, turned into left minus right, then the
                    # two sides recovered from the mean and the difference.
                    diff = sign * float(row[at[versus_field]])
                    rows[label]["left"].append(mean + n_right * diff / per_match)
                    rows[label]["right"].append(mean - n_left * diff / per_match)
                finished += 1
                if finished % 8 == 0 or finished == args.episodes:
                    kills_left = np.mean(rows["kills"]["left"])
                    kills_right = np.mean(rows["kills"]["right"])
                    print(
                        f"{finished:4d}/{args.episodes} episodes | kills a match "
                        f"left {kills_left:5.2f} right {kills_right:5.2f} | "
                        f"{steps * slots / (time.perf_counter() - started):,.0f} steps/s",
                        flush=True,
                    )
    finally:
        pool.close()

    result = {
        "left": left.name,
        "right": right.name,
        "episodes": finished,
        "agents": agents,
        "front": front,
        "back": back,
        "maps": layout.maps,
        "greedy": args.greedy,
        "seed": args.seed,
        "metrics": {},
    }
    print()
    print(f"{'':18s}{'left':>9s}{'right':>9s}{'left - right':>15s}{'95% interval':>22s}")
    for label, _, _ in METRICS:
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
    kills = result["metrics"]["kills"]
    low, high = kills["interval"]
    if low > 0:
        verdict = f"{left.name} is ahead on kills"
    elif high < 0:
        verdict = f"{right.name} is ahead on kills"
    else:
        verdict = "no difference on kills that these episodes can tell apart"
    result["verdict"] = verdict
    print()
    print(verdict, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"written to {args.json}")
    return result


if __name__ == "__main__":
    main()
