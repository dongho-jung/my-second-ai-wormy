"""Self-play PPO over the headless WebLiero environment.

Every worm in every match is the same policy, so the opponent improves exactly
as fast as the agent does and there is nothing to hand-write an enemy for. What
it is paid for lives in `src/env/reward.js`: hurting the others, killing them,
staying alive, and getting somewhere rather than sitting in a hole.

Run it and watch it on the monitor page:

    npm run train -- --total-steps 2000000 --label "three-way"
    npm run monitor
"""

from __future__ import annotations

import argparse
import copy
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demos as demo_store
from policy import WormPolicy
from run import Run
from workers import REPO, WorkerPool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    world = parser.add_argument_group("the match")
    world.add_argument("--agents", type=int, default=3, help="worms per match: 1 solo, 2 a duel, 5 a brawl")
    world.add_argument("--observation-foes", type=int, default=None,
                       help="size the vector for this many other worms, so one policy can play any count")
    world.add_argument("--episode-ticks", type=int, default=3600)
    world.add_argument("--frameskip", type=int, default=4)
    world.add_argument("--patch-scale", type=int, default=2,
                       help="pixels per patch cell. The patch shows the same ground at any "
                            "scale: 2 is 213x121 cells, 4 is 107x61 and a quarter of the "
                            "convolution, which is most of what an update costs. Baked into "
                            "the checkpoint, so the viewer plays back at the trained scale")
    world.add_argument("--input-latency", type=str, default="6-21",
                       help="ticks between deciding and acting. The room this plays in runs about "
                            "300ms behind, which is eighteen ticks; training at the old 0-3 taught "
                            "reflexes that do not survive the trip. A range rather than a number "
                            "so the policy does not learn one particular delay")
    world.add_argument("--maps", type=int, default=0,
                       help="generated dirt levels to mix in. Off: the room this trains for plays "
                            "community maps, and a generated field is a map it never picks. "
                            "Pass 12 to mix them back in")
    world.add_argument("--map-width", type=int, default=504,
                       help="the width the game itself generates; narrower makes worms meet sooner")
    world.add_argument("--levels-dir", default=str(REPO / "artifacts" / "maps" / "dsds-cs"),
                       help="the maps to play on. Defaults to the cs_ pool `npm run maps` "
                            "downloads, which is what the watched room runs. The game's own "
                            "pool is artifacts/levels, where `npm run levels` puts it — it was "
                            "the default here while the cs_ maps sat in another directory "
                            "entirely, so every run that meant to train on them trained on "
                            "generated dirt and the stock pool instead")
    world.add_argument("--stock-levels", type=int, default=64,
                       help="how many of them to use; 0 trains on generated maps alone")
    world.add_argument("--drill", action="store_true",
                       help="give every worm five of one weapon, a different one each episode. "
                            "A worm handed five unfamiliar weapons in a fight learns nothing "
                            "about any of them; one for a whole episode is long enough to find "
                            "out what it does")
    world.add_argument("--weapons", default="room", choices=["starter", "room", "direct", "all"],
                       help="which weapons a worm can spawn holding. `starter` is a hand-picked 45 for "
                            "learning to aim before learning what a rocket does to whoever "
                            "fired it; `room` is the list the "
                            "watched room actually allows, read off its weapon screen: 95 of the "
                            "129, with the other 34 reachable only out of a crate")
    world.add_argument("--allow-unmeasured-weapons", action="store_true",
                       help="start even though the mod's weapons have not been measured "
                            "(`npm run weapons`). Without the profile the policy sees no "
                            "weapon behaviour — 72 fields of the vector stay zero — and the "
                            "aim rewards pay nothing. The first cluster run went two and a "
                            "half hours that way. Only for a deliberate experiment")
    world.add_argument("--mod", default=None,
                       help="the game to train under. Defaults to whichever mod the environment "
                            "does, so the name does not live in two languages at once — it was "
                            "spelled here as well, went stale, and the run duly ignored every "
                            "recording from the room it was meant to be learning from")
    world.add_argument("--ban-start", default="",
                       help="further weapons nobody spawns holding, by name, on top of whatever "
                            "--weapons already allows. Names are ambiguous where a mod uses one "
                            "twice, so prefer --weapons room, which goes by position")
    world.add_argument("--rules", default="room", choices=["room", "clean"],
                       help="room matches the watched room's own settings, read off it: weapon "
                            "crates every 480 ticks and a weapon-change delay. The engine's bare "
                            "defaults drop health only, so a weapon barred from the loadout would "
                            "never be seen at all. clean turns bonuses off")
    world.add_argument("--no-patch", action="store_true", help="drop the close terrain patch")
    world.add_argument("--no-map", action="store_true",
                       help="drop the whole-level picture. Without it a policy can climb the ledge in "
                            "front of it and has no way to decide which direction the match is in")

    scale = parser.add_argument_group("how much of it to run")
    # Measured on this machine: the environment costs almost nothing (6% of the
    # CPU) and the gradient steps on the GPU are the whole of it. So the batch
    # is collected from many worms over a short horizon rather than few worms
    # over a long one, and it is reused twice rather than four times — fresh
    # data is nearly free and reuse is not. 576 worms over 24 steps at four
    # epochs was 10,600 steps/s; this is 19,700.
    scale.add_argument("--workers", type=int, default=8)
    scale.add_argument("--envs", type=int, default=12, help="matches per worker")
    scale.add_argument("--steps", type=int, default=128,
                       help="decisions per worm per update. This is how far ahead a reward can "
                            "be and still reach the action that earned it: GAE only sees what "
                            "is inside one rollout, and past the end it has to trust the value "
                            "head instead. At 12 — 0.8 seconds — a grenade, a mine or a long "
                            "shot landed in the next rollout, after the action that threw it "
                            "had been learned from and discarded. 128 is eight and a half "
                            "seconds. The buffers cost steps x worms, so raising this and "
                            "leaving --envs alone is how a run runs out of memory")
    scale.add_argument("--bptt", type=int, default=32,
                       help="decisions per gradient chunk through the memory. Separate from "
                            "--steps: the rollout is collected and valued whole, then replayed "
                            "in chunks this long, each starting from the hidden state the "
                            "rollout itself had there. Long rollouts do not have to mean "
                            "backpropagating through all of them")
    scale.add_argument("--total-steps", type=int, default=2_000_000)

    learn = parser.add_argument_group("learning")
    learn.add_argument("--lr", type=float, default=3e-4)
    learn.add_argument("--target-kl", type=float, default=0.012,
                       help="how far the policy should move per update. Measured on a 63M-step run: "
                            "at a fixed rate it settled at a KL of 0.001, a tenth of what PPO normally "
                            "does, and the combat numbers went flat while it crawled. The rate is "
                            "nudged to hit this instead. 0 turns it off")
    learn.add_argument("--lr-range", type=str, default="1e-5,1e-3",
                       help="how far the rate may be nudged")
    learn.add_argument("--gamma", type=float, default=0.99)
    learn.add_argument("--lam", type=float, default=0.95)
    learn.add_argument("--clip", type=float, default=0.2)
    learn.add_argument("--epochs", type=int, default=2)
    learn.add_argument("--minibatches", type=int, default=4)
    # Sized against the policy loss rather than picked off a paper. With
    # normalised advantages the policy loss here runs around 0.0005, not the
    # 0.01-0.05 the usual 0.01 coefficient assumes — so that coefficient paid
    # the policy fifty-six times more to stay undecided than to win, and a run
    # sat at 80%% of maximum entropy for twenty million steps without ever
    # committing to anything. These keep the bonus near the size of the thing
    # it is competing with.
    learn.add_argument("--entropy", type=float, default=0.0001,
                       help="how much it is paid to stay undecided, at the start")
    learn.add_argument("--entropy-final", type=float, default=0.00001,
                       help="and at the end")
    league = parser.add_argument_group("who it plays")
    league.add_argument("--opponents", type=float, default=0.0,
                        help="what share of the worms are driven by an older copy of the "
                             "policy instead of the one being trained. Everybody being the "
                             "same policy means the average reward goes up when all three "
                             "get more reckless together, and there is no way to tell that "
                             "from getting better. 0.34 makes one worm in a three-way an "
                             "opponent. 0 is self-play as it has been")
    league.add_argument("--pool-size", type=int, default=6,
                        help="how many past copies to keep. They are sampled per rollout, so "
                             "a policy has to stay good against what it used to be rather "
                             "than against what it is right now")
    league.add_argument("--pool-every", type=int, default=25,
                        help="updates between adding the current policy to the pool")

    learn.add_argument("--shaping-decay", type=float, default=0.0,
                       help="what share of the run to fade the ladder rewards over. The aim, "
                            "approach and exploration terms exist to get a policy started — "
                            "aiming pays nothing in this game, and the payoff for it arrives "
                            "much later as damage — but each is also a way to score without "
                            "playing well, and none of them comes down on its own. 0.6 fades "
                            "them out over the first three fifths and leaves the rest of the "
                            "run on damage, kills and deaths. 0 keeps them, which is what "
                            "every run so far has done")
    learn.add_argument("--shaping-floor", type=float, default=0.0,
                       help="what the ladder is worth once it has faded; 0 is nothing at all")
    learn.add_argument("--value-coef", type=float, default=0.5)
    learn.add_argument("--max-grad-norm", type=float, default=0.5)
    learn.add_argument("--seed", type=int, default=1)

    shown = parser.add_argument_group("learning from recorded play")
    shown.add_argument("--demos", default=str(REPO / "artifacts" / "demos"),
                       help="recordings of people playing, as `npm run record` writes them")
    shown.add_argument("--bc-coef", type=float, default=0.005,
                       help="how much of each update is spent agreeing with the recordings, once "
                            "there are enough of them. The cloning loss is a sum of seven "
                            "cross-entropies; the policy loss, against normalised advantages, is "
                            "around 0.0005. At 0.5 the recordings pulled a hundred times harder "
                            "than the reward and a run memorised 2,571 frames to 99.9%. At 0.05 "
                            "they pulled seven times harder, which is subtler and ends the same "
                            "way: 99.5% agreement with the recordings while the kills fell. "
                            "0 ignores them")
    shown.add_argument("--bc-full-frames", type=int, default=20_000,
                       help="how many frames of play the coefficient above is worth in full. Below "
                            "it the weight is scaled down in proportion: a few minutes of somebody "
                            "finding their feet is a hint, not an authority")
    shown.add_argument("--bc-idle-share", type=float, default=0.2,
                       help="most of a recording is frames with nothing pressed; thin them to this")
    shown.add_argument("--bc-max-frames", type=int, default=30_000,
                       help="how many recorded frames to hold at once. Each carries the worm's "
                            "whole 426x240 view at thirty kilobytes, and they all live on the "
                            "training device together — an unbounded pile is what ran this "
                            "machine out of memory, at 227,592 frames and 6.9GB")
    shown.add_argument("--bc-rescan", type=int, default=40,
                       help="updates between re-reading the directory, so a match played now is "
                            "learned from without restarting anything")

    where = parser.add_argument_group("where it goes")
    where.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    where.add_argument("--label", default=None, help="a name for this run on the monitor page")
    where.add_argument("--resume", default=None,
                       help="a .pt to carry on from, so changing the layout does not throw away what it learned")
    where.add_argument("--save-every", type=int, default=20, help="updates between checkpoints")
    where.add_argument("--torch-threads", type=int, default=2,
                       help="more than a couple is slower here, and the cores are wanted by the workers")
    return parser.parse_args(argv)


# The figures carried across updates that ended with no episode finished.
SHOWN = ("episodeReward", "kills", "deaths", "damageDealt", "selfDamage", "stuckSteps")


LEVEL_SUFFIXES = (".lev", ".png")


def stock_levels(args):
    """The map files to play on, so training sees the maps a room actually picks."""
    if args.stock_levels <= 0:
        return []
    # The pool spells them both ways — BATTLEGR.LEV next to SMB.lev — and a
    # case-sensitive glob quietly drops six of the fifty-two. Community pools
    # publish PNGs instead, which are the same terrain in another wrapper.
    directory = Path(args.levels_dir)
    found = sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() in LEVEL_SUFFIXES and path.is_file()
    )[: args.stock_levels]
    if not found:
        print(f"no maps in {args.levels_dir}: training on generated ones alone", flush=True)
    return [str(path) for path in found]


def save(policy, layout, shape, step, path, **extra):
    """A checkpoint carries the shape of what it expects, so a viewer can load it
    without being told how the run was configured."""
    torch.save(
        {
            "policy": policy.state_dict(),
            "layout": {
                "vectorSize": layout.vector_size,
                "headSizes": layout.head_sizes,
                **shape,
                "agents": layout.agents,
                "frameskip": layout.frameskip,
                "episodeTicks": layout.episode_ticks,
            },
            "step": step,
            **extra,
        },
        path,
    )


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = pick_device(args.device)

    latency = [int(part) for part in str(args.input_latency).split("-")]
    config = dict(
        envs=args.envs,
        agents=args.agents,
        episodeTicks=args.episode_ticks,
        frameskip=args.frameskip,
        patchScale=args.patch_scale,
        # Every world starts its first episode partway through, so the matches
        # end at different times instead of all together every seven updates.
        stagger=True,
        inputLatencyTicks=latency if len(latency) > 1 else latency[0],
        levelPool=args.maps,
        levelFiles=stock_levels(args),
        levelOptions={"width": args.map_width},
        weaponPool=args.weapons,
        loadout="drill" if args.drill else "random",
        # The ladder's fade, in decisions one worm takes. The environment counts
        # its own steps and knows nothing about how many worms are playing or
        # how long the run is; both of those are here, so the arithmetic is too.
        # Filled in below, once the worker count is known.
        shapingFullAt=0,
        shapingFloor=args.shaping_floor,
        banStart=[name.strip() for name in args.ban_start.split(",") if name.strip()],
        # Read off the room this project watches, rather than assumed: it drops
        # weapon crates only, eight seconds apart, and makes a swapped-to weapon
        # wait three quarters of a second before it will fire.
        rules=(
            {"bonusDrops": 3, "bonusSpawnTicks": 480, "weaponChangeDelay": 45}
            if args.rules == "room"
            else {"bonusDrops": 0}
        ),
        seed=args.seed,
        **({"engine": {"mod": args.mod}} if args.mod else {}),
        observations=[
            "vector",
            *([] if args.no_patch else ["patchBytes"]),
            *([] if args.no_map else ["map"]),
        ],
    )
    if args.observation_foes is not None:
        config["observationFoes"] = args.observation_foes

    # How many worms at the back of each match are older copies. Worked out
    # before the workers start, because they have to be told: otherwise every
    # figure they report is averaged over both sides and the one number that
    # says whether this is improving averages itself away.
    frozen_per_match = min(args.agents - 1, int(round(args.agents * args.opponents)))
    config["opponents"] = frozen_per_match

    if args.shaping_decay > 0:
        # `--total-steps` counts every worm's decision; the environment counts
        # only its own. One worm's share of the run is the whole thing divided
        # by how many are playing it.
        worms = max(1, args.workers * args.envs * args.agents)
        config["shapingFullAt"] = int(args.total_steps * args.shaping_decay / worms)

    pool = WorkerPool(args.workers, config)
    layout = pool.layout
    slots = pool.slots
    if not layout.weapons_measured and not args.allow_unmeasured_weapons:
        pool.close()
        raise RuntimeError(
            f"the weapons of {layout.mod} have not been measured: there is no "
            "artifacts/weapons.<mod>.json, so the policy would see no weapon "
            "behaviour and the aim rewards would pay nothing. Run `npm run weapons` "
            "(and, for an image, make sure the file is in the build). "
            "--allow-unmeasured-weapons overrides this for a deliberate experiment"
        )
    use_patch = not args.no_patch and layout.patch_cells > 0
    use_map = not args.no_map and layout.map_cells > 0
    patch_shape = tuple(layout.patch_shape[1:]) if layout.patch_shape else (121, 213)
    map_side = layout.map_shape[1] if layout.map_shape else 32

    policy = WormPolicy(
        layout.vector_size,
        layout.head_sizes,
        patch_shape=patch_shape,
        use_patch=use_patch,
        use_map=use_map,
        map_side=map_side,
        weapon_ids_at=layout.weapon_ids_at,
        weapon_ids_count=layout.weapon_ids_count,
        weapon_count=layout.weapon_count,
    ).to(device)
    # What a checkpoint has to carry for a viewer or a resume to rebuild it.
    shape_of = {
        # The whole world this policy learned in, verbatim — not a hand-picked
        # few of its settings. Every time one was added to training it had to be
        # threaded to the viewer by hand as well, and each time one was missed
        # the viewer quietly showed a different game: generated dirt instead of
        # the room's maps, weapons the policy was never given, five random guns
        # where training hands out one, no input delay where training has three
        # hundred milliseconds of it. Carrying the lot ends that class of bug.
        "world": {
            key: value
            for key, value in config.items()
            # Not the seed, or how many copies were run in parallel.
            if key not in ("seed", "envs")
        },
        "usePatch": use_patch,
        "patchShape": list(patch_shape),
        "weaponIdsAt": layout.weapon_ids_at,
        "weaponIdsCount": layout.weapon_ids_count,
        "weaponCount": layout.weapon_count,
        "useMap": use_map,
        "mapSide": map_side,
    }
    # Seven independent choices, so the most undecided a policy can be is the
    # sum of each head's own maximum, not one action space's worth.
    max_entropy = float(np.log(np.array(layout.head_sizes, dtype=np.float64)).sum())
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    learning_rate = args.lr
    lr_low, lr_high = (float(part) for part in args.lr_range.split(","))
    parameters = sum(p.numel() for p in policy.parameters())

    # Carrying on from a checkpoint. The rollout shape is free to change — how
    # many worlds run at once is about how the machine is used, not about what
    # the policy is — so only the observation has to still match.
    resumed_from = None
    resumed_at = 0
    if args.resume:
        carried = torch.load(args.resume, map_location="cpu", weights_only=False)
        shape = carried["layout"]
        if shape["vectorSize"] != layout.vector_size or shape["headSizes"] != layout.head_sizes:
            raise RuntimeError(
                f"{args.resume} was trained on a vector of {shape['vectorSize']} "
                f"with heads {shape['headSizes']}, and this run gives "
                f"{layout.vector_size} with heads {layout.head_sizes}. A policy "
                "cannot be carried across a change to what it sees or what it "
                "may do; start a fresh run"
            )
        seen = tuple(shape.get("patchShape") or ())
        if use_patch and seen and seen != patch_shape:
            raise RuntimeError(
                f"{args.resume} looked at a {seen[1]}x{seen[0]} patch and this run "
                f"shows a {patch_shape[1]}x{patch_shape[0]} one"
            )
        policy.load_state_dict(carried["policy"])
        resumed_from = args.resume
        resumed_at = int(carried.get("step", 0))
        print(f"carrying on from {args.resume} at {resumed_at:,} steps", flush=True)

    run = Run(
        label=args.label or f"{args.agents}-way self-play",
        meta={
            "policy": f"PPO self-play, {parameters/1e6:.2f}M parameters",
            "device": str(device),
            "agents": args.agents,
            "workers": args.workers,
            "matches": pool.envs,
            "parallelWorms": slots,
            "observation": (
                f"vector {layout.vector_size}"
                + (f" + patch {layout.patch_shape[0]}x{patch_shape[0]}x{patch_shape[1]}" if use_patch else "")
                + (f" + map 4x{map_side}x{map_side}" if use_map else "")
            ),
            "rolloutSteps": args.steps,
            "batch": args.steps * slots,
            "rolloutSteps": args.steps,
            "bptt": args.bptt,
            "frameskip": args.frameskip,
            "episodeTicks": args.episode_ticks,
            "inputLatencyTicks": args.input_latency,
            "maps": layout.maps,
            "stockMaps": layout.stock_maps,
            "mapWidth": args.map_width,
            "weapons": args.weapons,
            "rules": args.rules,
            "engineSha256": layout.engine_sha256,
            "mod": layout.mod,
            "lr": args.lr,
            "targetKL": args.target_kl,
            "bcCoef": args.bc_coef,
            # The page needs to know whether any worm is an older copy: without
            # one, none of its averages can say whether this is improving.
            "opponents": frozen_per_match,
            "poolSize": args.pool_size if frozen_per_match else 0,
            "shapingDecay": args.shaping_decay,
            "gamma": args.gamma,
            "clip": args.clip,
            "entropy": args.entropy,
            "entropyFinal": args.entropy_final,
            "resumedFrom": resumed_from,
            "resumedAt": resumed_at,
        },
    )
    print(f"run {run.id} -> {run.path}", flush=True)
    print(
        f"{slots} worms across {pool.envs} matches on {args.workers} workers, "
        f"{parameters/1e6:.2f}M parameters on {device}",
        flush=True,
    )
    run.note(
        f"{args.agents}-way self-play started: {slots} worms at once, "
        f"{args.steps * slots:,} steps per update"
    )

    # Which worms are somebody else.
    #
    # One policy playing itself cannot tell "I got better" from "we all got
    # more reckless at the same time": the number that goes up is the average
    # over the same weights. So some of the worms are driven by copies of what
    # this policy used to be. They act, and they are acted against, and nothing
    # is learned from their transitions — their actions did not come from the
    # policy being updated, so their log-probs are somebody else's.
    #
    # Taken from the end of each match's block of worms, so worm 0 of every
    # match is always the one being trained and a match is never all opponents.
    per_match = layout.agents
    is_opponent = torch.zeros(slots, dtype=torch.bool, device=device)
    # One block of lane indices per opponent seat, so each seat can be played by
    # a different generation.
    opponent_seats = []
    if frozen_per_match > 0:
        for seat in range(per_match - frozen_per_match, per_match):
            is_opponent[seat::per_match] = True
            opponent_seats.append(
                torch.arange(seat, slots, per_match, device=device)
            )
    learners = (~is_opponent).nonzero(as_tuple=True)[0]
    opponents = is_opponent.nonzero(as_tuple=True)[0]
    if frozen_per_match:
        print(
            f"{len(opponents)} of {slots} worms are older copies "
            f"({frozen_per_match} per {per_match}-worm match)",
            flush=True,
        )

    heads_count = len(layout.head_sizes)
    # What a `dones` byte means. `first` is the opening observation of a new
    # episode — the only place the memory is cleared. `last` is the closing one
    # of an old episode, which is worth what it is worth: the clock ran out, the
    # match did not. The action sampled at a `last` state is spent, because the
    # world restarts instead of applying it, so those steps are held out of the
    # update rather than learned from as though they had consequences.
    DONE_ONGOING = float(layout.done_codes["ongoing"])
    DONE_FIRST = float(layout.done_codes["first"])
    DONE_LAST = float(layout.done_codes["last"])
    obs_v = torch.zeros(args.steps, slots, layout.vector_size, device=device)
    obs_p = (
        torch.zeros(args.steps, slots, layout.patch_cells, dtype=torch.uint8, device=device)
        if use_patch
        else None
    )
    obs_m = (
        torch.zeros(args.steps, slots, layout.map_cells, dtype=torch.uint8, device=device)
        if use_map
        else None
    )
    acts = torch.zeros(args.steps, slots, heads_count, dtype=torch.long, device=device)
    logps = torch.zeros(args.steps, slots, device=device)
    vals = torch.zeros(args.steps, slots, device=device)
    rews = torch.zeros(args.steps, slots, device=device)
    dones = torch.zeros(args.steps, slots, device=device)
    # Where a worm came back from the dead, one per slot rather than one per
    # match. A match's memory restarts with the match; a worm's restarts here
    # too, because what it remembers is a life that is over.
    resets = torch.zeros(args.steps, slots, device=device)
    # What the policy was remembering when each step was decided, so the update
    # can replay the rollout from the same state the rollout actually saw.
    carried = torch.zeros(args.steps, slots, policy.memory_width, device=device)
    memory = torch.zeros(slots, policy.memory_width, device=device)

    vectors, patches, maps, _, _, _, _ = pool.observations()
    next_v = torch.as_tensor(vectors, device=device)
    next_p = torch.as_tensor(patches, device=device) if use_patch else None
    next_m = torch.as_tensor(maps, device=device) if use_map else None
    next_done = torch.zeros(slots, device=device)
    next_reset = torch.zeros(slots, device=device)

    # Recorded play, folded into every update rather than left in a file
    # nothing reads. Re-read as the run goes, so somebody playing right now is
    # being learned from within a minute.
    expect = {
        "mod": layout.mod,
        "vectorSize": layout.vector_size,
        "patchCells": layout.patch_cells if use_patch else 0,
        "mapCells": layout.map_cells if use_map else 0,
        "heads": len(layout.head_sizes),
    }
    shown = None
    shown_at = -1

    def reload_demos():
        nonlocal shown, shown_at
        found = (
            demo_store.load(Path(args.demos), expect=expect, limit=args.bc_max_frames)
            if args.bc_coef > 0
            else None
        )
        shown = found.thin_idle(args.bc_idle_share) if found else None
        shown_at = updates
        if shown is not None:
            return dict(
                vectors=torch.as_tensor(shown.vectors, device=device),
                patches=torch.as_tensor(shown.patches, device=device) if use_patch else None,
                maps=torch.as_tensor(shown.maps, device=device) if use_map else None,
                heads=torch.as_tensor(shown.heads.astype(np.int64), device=device),
                acting=shown.acting,
                count=len(shown),
            )
        return None

    finished = []          # episodes that ended since the last update
    # The last policy is not the best one: self-play wanders, and a run watched
    # afterwards should be the best it ever played, not wherever it happened to
    # stop. Kept on a smoothed episode reward so one lucky batch cannot win it.
    best_reward = None
    smoothed = None
    # An episode is longer than a rollout, so most updates end with none of them
    # finished. The last numbers stay on the line rather than reading as zero.
    latest = {}
    # Counted from where the checkpoint left off, so the charts continue rather
    # than starting again at zero.
    total_steps = resumed_at
    updates = 0
    demo_batch = reload_demos()

    def demo_weight():
        """Trust the recordings in proportion to how many there are."""
        if not demo_batch or args.bc_coef <= 0:
            return 0.0
        return args.bc_coef * min(1.0, demo_batch["acting"] / max(1, args.bc_full_frames))

    if demo_batch:
        print(
            f"learning from {demo_batch['count']:,} recorded frames "
            f"({demo_batch['acting']:,} with something pressed), "
            f"at weight {demo_weight():.4f} of {args.bc_coef}",
            flush=True,
        )
    # Copies of what this policy used to be, and a network per opponent seat to
    # play them back through. A state dict is 1.6M floats — six of them is under
    # 40MB, so they are kept in memory rather than read off disk every rollout.
    #
    # One network per seat rather than one for all of them, because a match with
    # three opponents in it should be three different opponents. A field of
    # identical copies is one opponent standing in three places, and beating it
    # says less than beating a spread of what this policy used to be.
    frozen = []
    past = []
    if frozen_per_match:
        for _ in range(frozen_per_match):
            copy_of = copy.deepcopy(policy).to(device)
            for parameter in copy_of.parameters():
                parameter.requires_grad_(False)
            copy_of.eval()
            frozen.append(copy_of)
        # Seeded with the policy as it starts out. An opponent that presses keys
        # at random is a low bar, but it is a fixed one, which is the point.
        past.append({k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})

    def draw_opponents():
        """A generation for each opponent seat, different ones where there are
        enough to go round."""
        if not past:
            return
        spread = (
            random.sample(past, len(frozen))
            if len(past) >= len(frozen)
            else [random.choice(past) for _ in frozen]
        )
        for network, weights in zip(frozen, spread):
            network.load_state_dict(weights)

    started = time.perf_counter()
    # Every worm's observations still flatten together: they feed the running
    # normaliser, and an older copy's view of the world is a real one. What
    # narrows is the set of lanes an update reads from.
    lane_pool = learners
    batch = args.steps * slots
    lanes_per_batch = max(1, len(lane_pool) // args.minibatches)
    span = max(1, min(args.bptt, args.steps))
    # One update reads this many transitions at a time: a slice of the worms,
    # over one chunk of the rollout. The cloning loss draws the same number of
    # recorded frames, which is the proportion its weight was measured at.
    minibatch = max(1, lanes_per_batch * span)

    try:
        while total_steps - resumed_at < args.total_steps:
            # Exploration is worth paying for early and worth stopping later:
            # the bonus is a fixed size while the advantages are normalised, so
            # a coefficient that does not come down eventually outweighs
            # whatever the policy has learned and holds it at random.
            draw_opponents()
            bc_weight = demo_weight()
            progress = min(1.0, max(0.0, (total_steps - resumed_at) / max(1, args.total_steps)))
            entropy_coef = args.entropy + (args.entropy_final - args.entropy) * progress
            rollout_started = time.perf_counter()
            env_seconds = 0.0
            for step in range(args.steps):
                obs_v[step] = next_v
                if use_patch:
                    obs_p[step] = next_p
                if use_map:
                    obs_m[step] = next_m
                dones[step] = next_done
                resets[step] = next_reset
                carried[step] = memory
                restart = ((next_done == DONE_FIRST) | (next_reset > 0)).float()
                with torch.no_grad():
                    before = memory
                    head, logp, _, value, memory = policy.act(
                        next_v, next_p, next_m, want_entropy=False,
                        carried=before, restart=restart,
                    )
                    for network, seat in zip(frozen, opponent_seats):
                        # The same observation, through an older set of weights
                        # and an older memory. Its log-prob and value are the
                        # learner's and are wrong for these seats, which is why
                        # nothing is learned from them.
                        theirs, _, _, _, remembered = network.act(
                            next_v[seat],
                            next_p[seat] if use_patch else None,
                            next_m[seat] if use_map else None,
                            want_entropy=False,
                            carried=before[seat],
                            restart=restart[seat],
                        )
                        head[seat] = theirs
                        memory[seat] = remembered
                acts[step] = head
                logps[step] = logp
                vals[step] = value

                at = time.perf_counter()
                pool.step(head.to(torch.uint8).cpu().numpy())
                vectors, patches, maps, rewards, env_done, restarts, stats = pool.observations()
                env_seconds += time.perf_counter() - at

                rews[step] = torch.as_tensor(rewards, device=device)
                next_v = torch.as_tensor(vectors, device=device)
                if use_patch:
                    next_p = torch.as_tensor(patches, device=device)
                if use_map:
                    next_m = torch.as_tensor(maps, device=device)
                # A match ends for all of its worms at once.
                next_done = torch.as_tensor(
                    np.repeat(env_done, layout.agents).astype(np.float32), device=device
                )
                next_reset = torch.as_tensor(restarts.astype(np.float32), device=device)
                # Only where an episode just closed. The stats block is not
                # cleared between steps, so counting the restart byte as well
                # would file every episode twice, the second time from a buffer
                # that had already been read.
                closed = env_done == int(DONE_LAST)
                if closed.any():
                    finished.extend(stats[closed])
                total_steps += slots

            rollout_seconds_only = time.perf_counter() - rollout_started
            # Generalised advantage estimation, back through the rollout.
            with torch.no_grad():
                _, _, _, bootstrap, _ = policy.act(
                    next_v, next_p, next_m, want_entropy=False,
                    carried=memory,
                    restart=((next_done == DONE_FIRST) | (next_reset > 0)).float(),
                )
                advantages = torch.zeros_like(rews)
                running = torch.zeros(slots, device=device)
                for step in reversed(range(args.steps)):
                    if step == args.steps - 1:
                        ahead, flag = bootstrap, next_done
                    else:
                        ahead, flag = vals[step + 1], dones[step + 1]
                    # Two different questions, and they used to share one answer.
                    #
                    # "Is the state ahead worth anything?" — yes, unless it is
                    # the opening of a fresh episode, which is a state this
                    # action did not lead to. The closing state of an old
                    # episode is worth exactly what the value head says: the
                    # match was still going when this project stopped watching,
                    # and calling it worthless taught every worm that the world
                    # ends a minute in.
                    carry_value = (flag != DONE_FIRST).to(rews.dtype)
                    # "Does the advantage run back past here?" — only through
                    # the middle of an episode. Not across either boundary.
                    carry_run = (flag == DONE_ONGOING).to(rews.dtype)
                    delta = rews[step] + args.gamma * ahead * carry_value - vals[step]
                    running = delta + args.gamma * args.lam * carry_run * running
                    advantages[step] = running
                returns = advantages + vals
                # The steps whose action was never applied — and, if some worms
                # are older copies, everything they did as well.
                valid = (dones != DONE_LAST).to(rews.dtype)
                if frozen:
                    valid = valid * (~is_opponent).to(rews.dtype)

            flat_v = obs_v.reshape(batch, -1)
            flat_p = obs_p.reshape(batch, -1) if use_patch else None
            flat_m = obs_m.reshape(batch, -1) if use_map else None
            flat_a = acts.reshape(batch, heads_count)
            flat_logp = logps.reshape(batch)
            flat_adv = advantages.reshape(batch)
            flat_ret = returns.reshape(batch)
            flat_val = vals.reshape(batch)
            policy.norm.observe(flat_v)

            # Kept on the device and read once at the end: turning a loss into a
            # Python float waits for the GPU, and doing that five times per
            # minibatch is most of an update spent synchronising.
            if args.bc_coef > 0 and updates - shown_at >= args.bc_rescan:
                before = demo_batch["count"] if demo_batch else 0
                demo_batch = reload_demos()
                after = demo_batch["count"] if demo_batch else 0
                if after != before:
                    run.note(f"recorded play: {after:,} frames to learn from")
                    print(f"recordings now hold {after:,} frames", flush=True)
            names = ("policy", "value", "entropy", "clipped", "kl", "bc", "bcAgree")
            running_losses = torch.zeros(len(names), device=device)
            passes = 0
            # Whole worms, replayed in order, rather than a shuffle of single
            # steps. A memory only means anything in sequence: scoring step 12
            # of a match from a blank mind is scoring a different decision than
            # the one that was made. So a minibatch is a handful of worms and
            # every step they took, run through the memory the way it ran the
            # first time.
            # How far the gradient runs through the memory, which no longer has
            # to be how far the rollout runs. The rollout is collected and
            # valued whole — that is what lets a reward eight seconds later
            # reach the action that earned it — and then replayed in chunks this
            # long. Each chunk picks up the hidden state the rollout itself had
            # at its first step, so the memory is never scored from a blank mind
            # in the middle of a match.
            chunks = [(at, min(at + span, args.steps)) for at in range(0, args.steps, span)]
            for _ in range(args.epochs):
                order = lane_pool[torch.randperm(len(lane_pool), device=device)]
                for start in range(0, len(lane_pool), lanes_per_batch):
                    lanes = order[start : start + lanes_per_batch]
                    for first, stop in chunks:
                        kept = carried[first][lanes]
                        logp_steps, entropy_steps, value_steps = [], [], []
                        for step in range(first, stop):
                            _, lp, ent, val, kept = policy.act(
                                obs_v[step][lanes],
                                obs_p[step][lanes] if use_patch else None,
                                obs_m[step][lanes] if use_map else None,
                                acts[step][lanes],
                                carried=kept,
                                restart=(
                                    (dones[step][lanes] == DONE_FIRST) | (resets[step][lanes] > 0)
                                ).to(kept.dtype),
                            )
                            logp_steps.append(lp)
                            entropy_steps.append(ent)
                            value_steps.append(val)
                        logp = torch.cat(logp_steps)
                        entropy = torch.cat(entropy_steps)
                        value = torch.cat(value_steps)
                        # Flattened the same way the steps were concatenated.
                        take_logp = logps[first:stop, lanes].reshape(-1)
                        take_ret = returns[first:stop, lanes].reshape(-1)
                        # 1 for the steps whose action the world actually applied.
                        take_valid = valid[first:stop, lanes].reshape(-1)
                        counted = take_valid.sum().clamp(min=1.0)
                        ratio = (logp - take_logp).exp()
                        advantage = advantages[first:stop, lanes].reshape(-1)
                        # Centred and scaled over the steps that count, so a spent
                        # one cannot drag the whole minibatch's baseline with it.
                        mean = (advantage * take_valid).sum() / counted
                        var = ((advantage - mean).pow(2) * take_valid).sum() / counted
                        advantage = (advantage - mean) / (var.sqrt() + 1e-8)
                        unclipped = ratio * advantage
                        clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
                        policy_loss = -(torch.min(unclipped, clipped) * take_valid).sum() / counted
                        value_loss = 0.5 * ((value - take_ret).pow(2) * take_valid).sum() / counted
                        entropy_loss = (entropy * take_valid).sum() / counted
                        # Agreeing with what a person did, alongside being paid
                        # for the outcome. The reward says what is good; the
                        # recordings say what to try, which is the half exploration
                        # is worst at — nothing random ever throws the rope.
                        bc_loss = torch.zeros((), device=device)
                        bc_agree = torch.zeros((), device=device)
                        if demo_batch is not None:
                            pick = torch.randint(
                                0, demo_batch["count"], (min(minibatch, demo_batch["count"]),),
                                device=device,
                            )
                            # Single frames, each from a blank memory: a recording
                            # is not a sequence this policy ever lived through.
                            shown_logits, _, _ = policy(
                                demo_batch["vectors"][pick],
                                demo_batch["patches"][pick] if use_patch else None,
                                demo_batch["maps"][pick] if use_map else None,
                            )
                            target = demo_batch["heads"][pick]
                            bc_loss = sum(
                                nn.functional.cross_entropy(head, target[:, at])
                                for at, head in enumerate(shown_logits)
                            )
                            with torch.no_grad():
                                bc_agree = torch.stack(
                                    [
                                        (head.argmax(dim=1) == target[:, at]).to(torch.float32).mean()
                                        for at, head in enumerate(shown_logits)
                                    ]
                                ).mean()
                        loss = (
                            policy_loss
                            + args.value_coef * value_loss
                            - entropy_coef * entropy_loss
                            + bc_weight * bc_loss
                        )
                        optimiser.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                        optimiser.step()
                        with torch.no_grad():
                            running_losses += torch.stack(
                                (
                                    policy_loss.detach(),
                                    value_loss.detach(),
                                    entropy_loss.detach(),
                                    ((ratio - 1).abs() > args.clip).to(torch.float32).mean(),
                                    # The signed mean of the log-ratio is an
                                    # estimator that cancels itself out: a policy
                                    # that moved a long way in both directions
                                    # reports nearly zero, and a controller reading
                                    # it keeps raising the rate until the updates
                                    # are scrambling what was learned — which is
                                    # what happened, entropy climbing while the
                                    # bonus for it was being annealed away. This is
                                    # the standard non-negative estimator instead.
                                    ((ratio - 1) - (logp - take_logp)).mean(),
                                    bc_loss.detach(),
                                    bc_agree,
                                )
                            )
                        passes += 1

            updates += 1
            if frozen and updates % args.pool_every == 0:
                past.append({k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
                # Oldest out first, so the pool is a window on the recent past
                # rather than a museum of the first few minutes.
                while len(past) > max(1, args.pool_size):
                    past.pop(0)
            wall = time.perf_counter() - started
            rollout_seconds = time.perf_counter() - rollout_started
            losses = dict(zip(names, (running_losses / passes).tolist()))
            # Keep the size of an update honest. Too small and it learns almost
            # nothing per sample however many samples it sees; too large and it
            # falls off the cliff PPO's clipping exists to avoid.
            if args.target_kl > 0:
                kl = abs(losses["kl"])
                scale = 1.0
                if kl < args.target_kl / 1.5:
                    scale = 1.02
                elif kl > args.target_kl * 1.5:
                    scale = 1 / 1.02
                if scale != 1.0:
                    learning_rate = min(lr_high, max(lr_low, learning_rate * scale))
                    for group in optimiser.param_groups:
                        group["lr"] = learning_rate
            scored_ret = returns[:, lane_pool].reshape(-1)
            scored_val = vals[:, lane_pool].reshape(-1)
            variance = float(scored_ret.var())
            explained = (
                0.0 if variance == 0
                else 1 - float((scored_ret - scored_val).var()) / variance
            )
            episodes = np.array(finished) if finished else None
            finished = []
            line = dict(
                step=total_steps,
                update=updates,
                elapsedSeconds=wall,
                stepsPerSecond=(args.steps * slots) / rollout_seconds,
                # Where an update's time went: waiting on the workers, deciding
                # and storing the rollout, and the gradient steps themselves.
                envShare=env_seconds / rollout_seconds,
                rolloutShare=rollout_seconds_only / rollout_seconds,
                policyLoss=losses["policy"],
                valueLoss=losses["value"],
                entropy=losses["entropy"],
                # The same thing as a share of "every key equally likely". The
                # raw figure means nothing without knowing there are seven
                # decisions in an action; this reads straight as "still mashing".
                entropyShare=losses["entropy"] / max_entropy,
                entropyCoef=entropy_coef,
                clipFraction=losses["clipped"],
                approxKL=losses["kl"],
                learningRate=learning_rate,
                bcLoss=losses["bc"],
                # How often it would press what the person pressed. This is the
                # number that says whether playing a match made any difference.
                demoAgreement=losses["bcAgree"],
                demoFrames=demo_batch["count"] if demo_batch else 0,
                demoWeight=bc_weight,
                explainedVariance=explained,
                meanReward=float(rews.mean()) * args.steps,
            )
            if episodes is not None:
                for index, field in enumerate(layout.stat_fields):
                    if field == "seed":
                        continue
                    name = {"reward": "episodeReward", "steps": "episodeSteps"}.get(field, field)
                    line[name] = float(episodes[:, index].mean())
                line["episodes"] = int(len(episodes))
                line["damageRatio"] = (
                    line["damageDealt"] / line["damageTaken"] if line["damageTaken"] > 0 else 0.0
                )
            run.record(**line)
            latest.update({key: value for key, value in line.items() if key in SHOWN})
            print(
                f"update {updates:4d} | {total_steps:>10,} steps | "
                f"{line['stepsPerSecond']:>7,.0f}/s | "
                # Where the update's time went, so a slow run can be read
                # straight from the log instead of guessed at.
                f"sim {line['envShare']:>3.0%} act "
                f"{line['rolloutShare'] - line['envShare']:>3.0%} learn "
                f"{1 - line['rolloutShare']:>3.0%} | reward "
                f"{latest.get('episodeReward', float('nan')):7.3f} | "
                f"k/d {latest.get('kills', 0):.2f}/{latest.get('deaths', 0):.2f} | "
                f"dealt {latest.get('damageDealt', 0):6.1f} self {latest.get('selfDamage', 0):6.1f} | "
                f"stuck {latest.get('stuckSteps', 0):5.1f} | entropy {line['entropy']:.2f}",
                flush=True,
            )
            if "episodeReward" in line:
                smoothed = (
                    line["episodeReward"]
                    if smoothed is None
                    else 0.9 * smoothed + 0.1 * line["episodeReward"]
                )
                if best_reward is None or smoothed > best_reward:
                    best_reward = smoothed
                    save(policy, layout, shape_of, total_steps, run.path / "best.pt",
                         reward=best_reward)
                    run.record(step=total_steps, bestReward=best_reward)
            if updates % args.save_every == 0:
                save(policy, layout, shape_of, total_steps, run.path / "policy.pt")
    except KeyboardInterrupt:
        run.note("stopped by hand")
        run.close(status="stopped", steps=total_steps)
        raise
    finally:
        pool.close()

    save(policy, layout, shape_of, total_steps, run.path / "policy.pt")
    run.note(f"finished: {total_steps:,} steps over {updates} updates")
    run.close(
        status="done",
        steps=total_steps,
        updates=updates,
        bestReward=best_reward,
    )


def _stop(*_):
    """Turn a container stop into the stop that Ctrl-C already handles.

    Kubernetes ends a container with SIGTERM, and Python's default handler
    ends the process without unwinding: neither the `except` nor the `finally`
    around the training loop runs, so the run's own file is left saying
    "running" long after nothing is running. Raising here takes the existing
    KeyboardInterrupt path, which closes the run as stopped.
    """
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    main()
