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
    world.add_argument("--input-latency", type=str, default="0-3",
                       help="ticks between deciding and acting, for the move to a live game")
    world.add_argument("--maps", type=int, default=12,
                       help="generated levels to mix in alongside the pool, as a random room would")
    world.add_argument("--map-width", type=int, default=504,
                       help="the width the game itself generates; narrower makes worms meet sooner")
    world.add_argument("--levels-dir", default=str(REPO / "artifacts" / "levels"),
                       help="the game's own level pool, as `npm run levels` downloads it")
    world.add_argument("--stock-levels", type=int, default=64,
                       help="how many of them to use; 0 trains on generated maps alone")
    world.add_argument("--weapons", default="room", choices=["starter", "room", "direct", "all"],
                       help="which weapons a worm can spawn holding. `starter` is a hand-picked 45 for "
                            "learning to aim before learning what a rocket does to whoever "
                            "fired it; `room` is the list the "
                            "watched room actually allows, read off its weapon screen: 95 of the "
                            "129, with the other 34 reachable only out of a crate")
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
    scale.add_argument("--envs", type=int, default=48, help="matches per worker")
    scale.add_argument("--steps", type=int, default=12, help="decisions per worm per update")
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
        inputLatencyTicks=latency if len(latency) > 1 else latency[0],
        levelPool=args.maps,
        levelFiles=stock_levels(args),
        levelOptions={"width": args.map_width},
        weaponPool=args.weapons,
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

    pool = WorkerPool(args.workers, config)
    layout = pool.layout
    slots = pool.slots
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
        # So a viewer replays the maps the policy knows rather than inventing
        # its own — the difference between watching it play and watching it
        # flounder somewhere it has never been.
        "levels": stock_levels(args),
        # The same for the weapons and the bonus rules: a viewer that spawns
        # worms holding the weapons training bars is showing a game nobody is
        # learning, and BARRACUDA turning up every round is how that looked.
        "weaponPool": args.weapons,
        "banStart": [name.strip() for name in args.ban_start.split(",") if name.strip()],
        "rules": args.rules,
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
                f"and this run gives {layout.vector_size}"
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

    heads_count = len(layout.head_sizes)
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

    vectors, patches, maps, _, _, _ = pool.observations()
    next_v = torch.as_tensor(vectors, device=device)
    next_p = torch.as_tensor(patches, device=device) if use_patch else None
    next_m = torch.as_tensor(maps, device=device) if use_map else None
    next_done = torch.zeros(slots, device=device)

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
    started = time.perf_counter()
    batch = args.steps * slots
    minibatch = max(1, batch // args.minibatches)

    try:
        while total_steps - resumed_at < args.total_steps:
            # Exploration is worth paying for early and worth stopping later:
            # the bonus is a fixed size while the advantages are normalised, so
            # a coefficient that does not come down eventually outweighs
            # whatever the policy has learned and holds it at random.
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
                with torch.no_grad():
                    head, logp, _, value = policy.act(
                        next_v, next_p, next_m, want_entropy=False
                    )
                acts[step] = head
                logps[step] = logp
                vals[step] = value

                at = time.perf_counter()
                pool.step(head.to(torch.uint8).cpu().numpy())
                vectors, patches, maps, rewards, env_done, stats = pool.observations()
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
                if env_done.any():
                    finished.extend(stats[env_done.astype(bool)])
                total_steps += slots

            rollout_seconds_only = time.perf_counter() - rollout_started
            # Generalised advantage estimation, back through the rollout.
            with torch.no_grad():
                _, _, _, bootstrap = policy.act(next_v, next_p, next_m, want_entropy=False)
                advantages = torch.zeros_like(rews)
                running = torch.zeros(slots, device=device)
                for step in reversed(range(args.steps)):
                    if step == args.steps - 1:
                        ahead, keep = bootstrap, 1.0 - next_done
                    else:
                        ahead, keep = vals[step + 1], 1.0 - dones[step + 1]
                    delta = rews[step] + args.gamma * ahead * keep - vals[step]
                    running = delta + args.gamma * args.lam * keep * running
                    advantages[step] = running
                returns = advantages + vals

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
            for _ in range(args.epochs):
                order = torch.randperm(batch, device=device)
                for start in range(0, batch, minibatch):
                    take = order[start : start + minibatch]
                    _, logp, entropy, value = policy.act(
                        flat_v[take],
                        flat_p[take] if use_patch else None,
                        flat_m[take] if use_map else None,
                        flat_a[take],
                    )
                    ratio = (logp - flat_logp[take]).exp()
                    advantage = flat_adv[take]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (value - flat_ret[take]).pow(2).mean()
                    entropy_loss = entropy.mean()
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
                        shown_logits, _ = policy(
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
                                ((ratio - 1) - (logp - flat_logp[take])).mean(),
                                bc_loss.detach(),
                                bc_agree,
                            )
                        )
                    passes += 1

            updates += 1
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
            variance = float(flat_ret.var())
            explained = 0.0 if variance == 0 else 1 - float((flat_ret - flat_val).var()) / variance
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
                f"{line['stepsPerSecond']:>7,.0f}/s | reward "
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


if __name__ == "__main__":
    main()
