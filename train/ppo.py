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
    world.add_argument("--weapons", default="all", choices=["direct", "all"],
                       help="a real room lets a player pick any of the forty, explosives included")
    world.add_argument("--rules", default="room", choices=["room", "clean"],
                       help="room matches the engine's own defaults, bonus drops and all")
    world.add_argument("--no-patch", action="store_true", help="vector observation only, about ten times cheaper")

    scale = parser.add_argument_group("how much of it to run")
    scale.add_argument("--workers", type=int, default=4)
    scale.add_argument("--envs", type=int, default=8, help="matches per worker")
    scale.add_argument("--steps", type=int, default=128, help="decisions per worm per update")
    scale.add_argument("--total-steps", type=int, default=2_000_000)

    learn = parser.add_argument_group("learning")
    learn.add_argument("--lr", type=float, default=3e-4)
    learn.add_argument("--gamma", type=float, default=0.99)
    learn.add_argument("--lam", type=float, default=0.95)
    learn.add_argument("--clip", type=float, default=0.2)
    learn.add_argument("--epochs", type=int, default=4)
    learn.add_argument("--minibatches", type=int, default=4)
    learn.add_argument("--entropy", type=float, default=0.01)
    learn.add_argument("--value-coef", type=float, default=0.5)
    learn.add_argument("--max-grad-norm", type=float, default=0.5)
    learn.add_argument("--seed", type=int, default=1)

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


def stock_levels(args):
    """The .lev files to mix in, so training sees the maps a room actually picks."""
    if args.stock_levels <= 0:
        return []
    # The pool spells them both ways — BATTLEGR.LEV next to SMB.lev — and a
    # case-sensitive glob quietly drops six of the fifty-two.
    directory = Path(args.levels_dir)
    found = sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() == ".lev" and path.is_file()
    )[: args.stock_levels]
    if not found:
        print(f"no .lev files in {args.levels_dir}: training on generated maps alone", flush=True)
    return [str(path) for path in found]


def save(policy, layout, use_patch, side, step, path, **extra):
    """A checkpoint carries the shape of what it expects, so a viewer can load it
    without being told how the run was configured."""
    torch.save(
        {
            "policy": policy.state_dict(),
            "layout": {
                "vectorSize": layout.vector_size,
                "headSizes": layout.head_sizes,
                "usePatch": use_patch,
                "patchSide": side,
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
        rules={} if args.rules == "room" else {"bonusDrops": 0},
        seed=args.seed,
        observations=["vector"] if args.no_patch else ["vector", "patchBytes"],
    )
    if args.observation_foes is not None:
        config["observationFoes"] = args.observation_foes

    pool = WorkerPool(args.workers, config)
    layout = pool.layout
    slots = pool.slots
    use_patch = not args.no_patch and layout.patch_cells > 0
    side = layout.patch_shape[1] if layout.patch_shape else 32

    policy = WormPolicy(
        layout.vector_size, layout.head_sizes, patch_side=side, use_patch=use_patch
    ).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
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
                f"vector {layout.vector_size}" + (f" + patch 4x{side}x{side}" if use_patch else "")
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
            "gamma": args.gamma,
            "clip": args.clip,
            "entropy": args.entropy,
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
    acts = torch.zeros(args.steps, slots, heads_count, dtype=torch.long, device=device)
    logps = torch.zeros(args.steps, slots, device=device)
    vals = torch.zeros(args.steps, slots, device=device)
    rews = torch.zeros(args.steps, slots, device=device)
    dones = torch.zeros(args.steps, slots, device=device)

    vectors, patches, _, _, _ = pool.observations()
    next_v = torch.as_tensor(vectors, device=device)
    next_p = torch.as_tensor(patches, device=device) if use_patch else None
    next_done = torch.zeros(slots, device=device)

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
    started = time.perf_counter()
    batch = args.steps * slots
    minibatch = max(1, batch // args.minibatches)

    try:
        while total_steps - resumed_at < args.total_steps:
            rollout_started = time.perf_counter()
            env_seconds = 0.0
            for step in range(args.steps):
                obs_v[step] = next_v
                if use_patch:
                    obs_p[step] = next_p
                dones[step] = next_done
                with torch.no_grad():
                    head, logp, _, value = policy.act(next_v, next_p, want_entropy=False)
                acts[step] = head
                logps[step] = logp
                vals[step] = value

                at = time.perf_counter()
                pool.step(head.to(torch.uint8).cpu().numpy())
                vectors, patches, rewards, env_done, stats = pool.observations()
                env_seconds += time.perf_counter() - at

                rews[step] = torch.as_tensor(rewards, device=device)
                next_v = torch.as_tensor(vectors, device=device)
                if use_patch:
                    next_p = torch.as_tensor(patches, device=device)
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
                _, _, _, bootstrap = policy.act(next_v, next_p, want_entropy=False)
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
            flat_a = acts.reshape(batch, heads_count)
            flat_logp = logps.reshape(batch)
            flat_adv = advantages.reshape(batch)
            flat_ret = returns.reshape(batch)
            flat_val = vals.reshape(batch)
            policy.norm.observe(flat_v)

            # Kept on the device and read once at the end: turning a loss into a
            # Python float waits for the GPU, and doing that five times per
            # minibatch is most of an update spent synchronising.
            names = ("policy", "value", "entropy", "clipped", "kl")
            running_losses = torch.zeros(len(names), device=device)
            passes = 0
            for _ in range(args.epochs):
                order = torch.randperm(batch, device=device)
                for start in range(0, batch, minibatch):
                    take = order[start : start + minibatch]
                    _, logp, entropy, value = policy.act(
                        flat_v[take], flat_p[take] if use_patch else None, flat_a[take]
                    )
                    ratio = (logp - flat_logp[take]).exp()
                    advantage = flat_adv[take]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (value - flat_ret[take]).pow(2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy * entropy_loss
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
                                (flat_logp[take] - logp).mean(),
                            )
                        )
                    passes += 1

            updates += 1
            wall = time.perf_counter() - started
            rollout_seconds = time.perf_counter() - rollout_started
            losses = dict(zip(names, (running_losses / passes).tolist()))
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
                clipFraction=losses["clipped"],
                approxKL=losses["kl"],
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
                    save(policy, layout, use_patch, side, total_steps, run.path / "best.pt",
                         reward=best_reward)
                    run.record(step=total_steps, bestReward=best_reward)
            if updates % args.save_every == 0:
                save(policy, layout, use_patch, side, total_steps, run.path / "policy.pt")
    except KeyboardInterrupt:
        run.note("stopped by hand")
        run.close(status="stopped", steps=total_steps)
        raise
    finally:
        pool.close()

    save(policy, layout, use_patch, side, total_steps, run.path / "policy.pt")
    run.note(f"finished: {total_steps:,} steps over {updates} updates")
    run.close(
        status="done",
        steps=total_steps,
        updates=updates,
        bestReward=best_reward,
    )


if __name__ == "__main__":
    main()
