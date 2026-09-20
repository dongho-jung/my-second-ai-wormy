"""Learn from recorded play, before learning from reward.

Reinforcement learning finds what random exploration stumbles onto. Some of
this game it will not stumble onto: throwing the rope, holding a direction,
shortening at the right moment and letting go is a sequence that pays nothing
until almost all of it is right. Being shown it once is worth a great many
episodes of not finding it.

So: plain supervised learning on what a person saw and what they pressed, into
the same network PPO then continues from.

    npm run clone                    # every demonstration recorded so far
    npm run train -- --resume artifacts/runs/<id>/clone.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy import WormPolicy
from run import DEFAULT_RUNS, Run
from workers import REPO

DEMOS = REPO / "artifacts" / "demos"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", default=str(DEMOS), help="directory of recordings")
    parser.add_argument("--only", default=None, help="one recording's id, instead of all of them")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--holdout", type=float, default=0.1, help="share kept back to measure on")
    parser.add_argument("--resume", default=None, help="carry on from a checkpoint instead of starting fresh")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-idle-share", type=float, default=0.2,
                        help="most of a recording is frames with nothing pressed — a player thinking, "
                             "or waiting. Left in, they are the majority class and cloning learns to "
                             "press nothing, which is exactly the habit this is meant to break")
    parser.add_argument("--min-samples", type=int, default=600,
                        help="refuse to train on less than this; a handful of frames teaches nothing")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    return parser.parse_args(argv)


def load(directory: Path, only: str | None):
    """Every recording that agrees about its shape, as one pile of samples."""
    metas = sorted(directory.glob("*.json"))
    if only:
        metas = [path for path in metas if path.stem == only]
    if not metas:
        raise FileNotFoundError(f"no recordings in {directory} — play a match with `npm run record` on")
    shape = None
    vectors, patches, maps, heads = [], [], [], []
    used = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        binary = meta_path.with_suffix(".bin")
        if not binary.exists():
            continue
        key = (meta["vectorSize"], meta["patchCells"], meta["mapCells"], len(meta["heads"]))
        if shape is None:
            shape = key
            first = meta
        elif key != shape:
            print(f"skipping {meta_path.stem}: recorded with a different observation", flush=True)
            continue
        size = meta["recordBytes"]
        raw = np.fromfile(binary, dtype=np.uint8)
        count = raw.size // size
        if count == 0:
            continue
        raw = raw[: count * size].reshape(count, size)
        at = 0
        vector_bytes = meta["vectorSize"] * 4
        vectors.append(raw[:, at : at + vector_bytes].copy().view(np.float32).reshape(count, -1))
        at += vector_bytes
        patches.append(raw[:, at : at + meta["patchCells"]].copy())
        at += meta["patchCells"]
        maps.append(raw[:, at : at + meta["mapCells"]].copy())
        at += meta["mapCells"]
        heads.append(raw[:, at : at + len(meta["heads"])].copy())
        used.append((meta_path.stem, count))
    if not vectors:
        raise RuntimeError("the recordings held no complete samples")
    return (
        first,
        np.concatenate(vectors),
        np.concatenate(patches),
        np.concatenate(maps),
        np.concatenate(heads),
        used,
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
    meta, vectors, patches, maps, heads, used = load(Path(args.demos), args.only)
    total = len(vectors)
    if total < args.min_samples:
        print(f"only {total} samples: not enough to learn anything. Play for longer.", flush=True)
        return 1
    # Thin out the frames where the player pressed nothing, so the ones where
    # they did something are what the network spends its capacity on.
    idle_mask = (heads == 0).all(axis=1)
    idle_count = int(idle_mask.sum())
    acting = int(total - idle_count)
    allowed = int(acting * args.max_idle_share / max(1e-6, 1 - args.max_idle_share))
    if idle_count > allowed and acting > 0:
        rng = np.random.default_rng(0)
        idle_index = np.flatnonzero(idle_mask)
        keep_idle = rng.choice(idle_index, size=allowed, replace=False)
        keep = np.sort(np.concatenate([np.flatnonzero(~idle_mask), keep_idle]))
        vectors, patches, maps, heads = (
            vectors[keep],
            patches[keep],
            maps[keep],
            heads[keep],
        )
        print(
            f"{total:,} samples were {idle_count/total*100:.0f}% nothing-pressed; "
            f"thinned to {len(keep):,} at {args.max_idle_share*100:.0f}%",
            flush=True,
        )
        total = len(keep)
    if acting < args.min_samples // 4:
        print(
            f"only {acting} frames with anything pressed. That is a recording of somebody "
            "standing still, and cloning it would teach standing still.",
            flush=True,
        )
        return 1
    device = pick_device(args.device)
    head_sizes = [head["choices"] for head in meta["heads"]]

    policy = WormPolicy(
        meta["vectorSize"],
        head_sizes,
        patch_side=32,
        use_patch=meta["patchCells"] > 0,
        use_map=meta["mapCells"] > 0,
        map_side=32,
    ).to(device)
    if args.resume:
        carried = torch.load(args.resume, map_location="cpu", weights_only=False)
        if carried["layout"]["vectorSize"] != meta["vectorSize"]:
            raise RuntimeError("that checkpoint was trained on a different observation")
        policy.load_state_dict(carried["policy"])
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)

    order = np.random.permutation(total)
    keep = int(total * (1 - args.holdout))
    train, test = order[:keep], order[keep:]
    to = lambda array, index, dtype: torch.as_tensor(array[index], dtype=dtype, device=device)
    # The observation normaliser learns its scale from the demonstrations, the
    # same as it would from a rollout.
    policy.norm.observe(to(vectors, train[: min(len(train), 20_000)], torch.float32))

    run = Run(
        label=args.label or "cloned from play",
        meta={
            "policy": f"behavioural cloning, {sum(p.numel() for p in policy.parameters())/1e6:.2f}M parameters",
            "device": str(device),
            "samples": total,
            "framesActing": acting,
            "heldOut": len(test),
            "recordings": ", ".join(f"{name} ({count})" for name, count in used),
            "observation": f"vector {meta['vectorSize']} + patch {meta['patchCells']} + map {meta['mapCells']}",
            "epochs": args.epochs,
            "lr": args.lr,
        },
        directory=Path(args.runs_dir),
    )
    print(f"run {run.id} -> {run.path}", flush=True)
    print(f"{total:,} samples from {len(used)} recording(s), {len(test):,} held back", flush=True)
    run.note(f"cloning {total:,} samples of recorded play")

    def evaluate(index):
        policy.eval()
        with torch.no_grad():
            logits, _ = policy(
                to(vectors, index, torch.float32),
                to(patches, index, torch.uint8) if meta["patchCells"] else None,
                to(maps, index, torch.uint8) if meta["mapCells"] else None,
            )
            target = to(heads, index, torch.long)
            agree = [
                (head.argmax(dim=1) == target[:, at]).float().mean().item()
                for at, head in enumerate(logits)
            ]
        policy.train()
        return agree

    best = None
    for epoch in range(args.epochs):
        np.random.shuffle(train)
        losses = 0.0
        batches = 0
        for start in range(0, len(train), args.batch):
            index = train[start : start + args.batch]
            logits, _ = policy(
                to(vectors, index, torch.float32),
                to(patches, index, torch.uint8) if meta["patchCells"] else None,
                to(maps, index, torch.uint8) if meta["mapCells"] else None,
            )
            target = to(heads, index, torch.long)
            loss = sum(
                nn.functional.cross_entropy(head, target[:, at])
                for at, head in enumerate(logits)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimiser.step()
            losses += float(loss)
            batches += 1
        agree = evaluate(test if len(test) else train[: args.batch])
        mean_agree = sum(agree) / len(agree)
        line = dict(
            step=(epoch + 1) * len(train),
            epoch=epoch + 1,
            cloneLoss=losses / max(1, batches),
            agreement=mean_agree,
        )
        for at, head in enumerate(meta["heads"]):
            line[f"agree_{head['name']}"] = agree[at]
        run.record(**line)
        print(
            f"epoch {epoch + 1:2d} | loss {line['cloneLoss']:.3f} | "
            f"agrees with the player {mean_agree * 100:.1f}% "
            f"({', '.join(f'{h['name']} {a*100:.0f}%' for h, a in zip(meta['heads'], agree))})",
            flush=True,
        )
        if best is None or mean_agree > best:
            best = mean_agree
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "layout": {
                        "vectorSize": meta["vectorSize"],
                        "headSizes": head_sizes,
                        "usePatch": meta["patchCells"] > 0,
                        "patchSide": 32,
                        "useMap": meta["mapCells"] > 0,
                        "mapSide": 32,
                        "agents": meta.get("foeSlots", 2) + 1,
                        "frameskip": 4,
                        "episodeTicks": 3600,
                    },
                    "step": line["step"],
                    "agreement": mean_agree,
                },
                run.path / "clone.pt",
            )
    run.note(f"done: agrees with the player {best * 100:.1f}% of the time")
    run.close(status="done", agreement=best, samples=total)
    print(f"\n-> {run.path / 'clone.pt'}", flush=True)
    print(f"   npm run train -- --resume {run.path / 'clone.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
