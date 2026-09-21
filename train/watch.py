"""Play one match with a trained policy, slowly enough to watch.

Loads a checkpoint, starts the viewer (`src/env/watch.js`), and plays. The
viewer serves the picture; this only decides what the worms do.

    npm run watch                       # the newest run's best policy
    npm run watch -- --agents 5 --speed 2
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy import WormPolicy
from run import DEFAULT_RUNS
from workers import REPO, _read_frame

VIEWER = REPO / "src" / "env" / "watch.js"


def newest_checkpoint(runs: Path) -> Path:
    """The best policy of the most recent run that saved one."""
    candidates = []
    for run in sorted(runs.glob("*/"), reverse=True):
        for name in ("best.pt", "policy.pt"):
            if (run / name).exists():
                candidates.append(run / name)
                break
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint under {runs} — train one first with: npm run train"
        )
    return candidates[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="run id to watch; the newest by default")
    parser.add_argument("--checkpoint", default=None, help="a .pt file, overriding --run")
    parser.add_argument("--agents", type=int, default=None,
                        help="worms in the match; the number it was trained with by default")
    parser.add_argument("--speed", type=float, default=1.0, help="1 is the speed the game runs at")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--levels", nargs="*", default=None,
                        help="map files to play on; by default the ones the checkpoint was trained on")
    parser.add_argument("--episode-ticks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true",
                        help="always take the most likely action instead of sampling")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="one match at a time is far too small to be worth a GPU")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = Path(args.runs_dir)
    if args.checkpoint:
        path = Path(args.checkpoint)
    elif args.run:
        run = runs / args.run
        path = run / "best.pt" if (run / "best.pt").exists() else run / "policy.pt"
    else:
        path = newest_checkpoint(runs)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    shape = checkpoint["layout"]
    trained_agents = shape.get("agents", 3)
    agents = args.agents or trained_agents
    device = torch.device(args.device)

    policy = WormPolicy(
        shape["vectorSize"],
        shape["headSizes"],
        patch_shape=tuple(shape.get("patchShape") or (121, 213)),
        weapon_ids_at=shape.get("weaponIdsAt"),
        weapon_ids_count=shape.get("weaponIdsCount", 0),
        weapon_count=shape.get("weaponCount", 0),
        use_patch=shape.get("usePatch", True),
        use_map=shape.get("useMap", False),
        map_side=shape.get("mapSide", 32),
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    # Whatever the policy actually learned in. The viewer adds only the things
    # that are about watching — where to serve it, how fast to run — and
    # overrides the rest only when asked on the command line.
    config = {
        **(shape.get("world") or {}),
        "agents": agents,
        # The vector keeps the width the policy was trained on however many
        # worms are actually playing, so a three-way policy can be watched in a
        # duel or a brawl without being retrained.
        "observationFoes": trained_agents - 1,
        "speed": args.speed,
        "port": args.port,
        "episodeTicks": args.episode_ticks or shape.get("episodeTicks", 3600),
    }
    if args.levels:
        config["levelFiles"] = [path for path in args.levels if Path(path).exists()]
    config["levelFiles"] = [
        path for path in (config.get("levelFiles") or []) if Path(path).exists()
    ]

    if args.seed is not None:
        config["seed"] = args.seed

    viewer = subprocess.Popen(
        ["node", str(VIEWER), json.dumps(config)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(REPO),
        bufsize=0,
    )
    try:
        layout = json.loads(_read_frame(viewer.stdout))
        if layout["vectorSize"] != shape["vectorSize"]:
            raise RuntimeError(
                f"the policy wants a vector of {shape['vectorSize']} and the match "
                f"gives {layout['vectorSize']}"
            )
        print(
            f"watching {path.parent.name} ({checkpoint.get('step', 0):,} steps"
            + (f", reward {checkpoint['reward']:.2f}" if "reward" in checkpoint else "")
            + f") — {layout['viewer']}",
            flush=True,
        )
        vector_bytes = layout["agents"] * layout["vectorSize"] * 4
        patch_bytes = layout["agents"] * layout["patchCells"]
        use_patch = shape.get("usePatch", True) and patch_bytes > 0
        map_bytes = layout["agents"] * layout.get("mapCells", 0)
        use_map = shape.get("useMap", False) and map_bytes > 0
        heads_count = len(layout["heads"])
        # Carried from one decision to the next: without this the policy is
        # handed a blank memory every frame and can hold nothing at all.
        carried = None
        while True:
            frame = _read_frame(viewer.stdout)
            vectors = torch.from_numpy(
                np.frombuffer(frame, dtype=np.float32, count=vector_bytes // 4).reshape(
                    layout["agents"], -1
                ).copy()
            ).to(device)
            patches = None
            if use_patch:
                patches = torch.from_numpy(
                    np.frombuffer(frame, dtype=np.uint8, count=patch_bytes, offset=vector_bytes)
                    .reshape(layout["agents"], -1)
                    .copy()
                ).to(device)
            maps = None
            if use_map:
                maps = torch.from_numpy(
                    np.frombuffer(
                        frame, dtype=np.uint8, count=map_bytes, offset=vector_bytes + patch_bytes
                    )
                    .reshape(layout["agents"], -1)
                    .copy()
                ).to(device)
            with torch.no_grad():
                if args.greedy:
                    logits, _, carried = policy(vectors, patches, maps, carried)
                    heads = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
                else:
                    heads, _, _, _, carried = policy.act(
                        vectors, patches, maps, want_entropy=False, carried=carried
                    )
            block = heads.to(torch.uint8).cpu().numpy().tobytes()
            viewer.stdin.write(struct.pack("<I", heads_count * layout["agents"]) + block)
            viewer.stdin.flush()
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    finally:
        try:
            viewer.stdin.close()
        except OSError:
            pass
        try:
            viewer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            viewer.kill()


if __name__ == "__main__":
    main()
