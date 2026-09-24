"""Play one match with a trained policy, slowly enough to watch.

Loads a checkpoint, starts the viewer (`src/env/watch.js`), and plays. The
viewer serves the picture; this only decides what the worms do.

    npm run watch                       # the newest run's best policy
    npm run watch -- --agents 5 --speed 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy import policy_from_shape
from movement_benchmark import fixed_movement_world
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
    parser.add_argument("--host", default=os.environ.get("WORMY_HOST", "127.0.0.1"),
                        help="what the viewer binds. Loopback by default; 0.0.0.0 in a container")
    parser.add_argument("--public-origin", default=os.environ.get("WORMY_PUBLIC_ORIGIN", ""),
                        help="where a browser actually reaches it, when that is not where it bound")
    parser.add_argument("--base-path", default=os.environ.get("WORMY_VIEWER_BASE_PATH", ""),
                        help="a path it is mounted under, such as /ai-worm/watch")
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
        # Not a failure so much as being early: a run writes its first
        # checkpoint after --save-every updates, and there is nothing to watch
        # until then. The monitor puts the last of stderr on the page, so this
        # says what to do rather than where it stopped.
        print(
            f"no checkpoint in {path.parent.name} yet — a run saves its first one "
            "after --save-every updates. Try again in a minute.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    shape = checkpoint["layout"]
    trained_agents = shape.get("agents", 3)
    agents = args.agents or trained_agents
    device = torch.device(args.device)

    policy = policy_from_shape(shape).to(device)
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
        # Where to serve it and what to answer to. The viewer prints the
        # address it can be reached at, and the monitor's Watch button opens
        # whatever it printed — so these are what make that button work from
        # somewhere that is not this machine.
        "host": args.host,
        "publicOrigin": args.public_origin or None,
        "basePath": args.base_path or "",
        "episodeTicks": args.episode_ticks or shape.get("episodeTicks", 3600),
    }
    if args.levels:
        config["levelFiles"] = [path for path in args.levels if Path(path).exists()]
    config["levelFiles"] = [
        path for path in (config.get("levelFiles") or []) if Path(path).exists()
    ]
    if not config["levelFiles"]:
        # A checkpoint from another machine names maps by paths that are not
        # here — a run from the cluster says /app/... — and with none of them
        # the viewer would play on one generated dirt field. Fall back to the
        # pool a fresh clone downloads, which is what the room plays.
        pool_dir = REPO / "artifacts" / "maps" / "dsds-cs"
        config["levelFiles"] = sorted(
            str(path) for path in pool_dir.glob("*")
            if path.suffix.lower() in (".lev", ".png") and path.is_file()
        )[:64]
        if config["levelFiles"]:
            print(
                f"the checkpoint's maps are not on this machine; playing the "
                f"{len(config['levelFiles'])} in {pool_dir}",
                flush=True,
            )

    # A movement Watch is the fixed benchmark made visible. Every coloured
    # worm is an independent simulation of the same checkpoint on the exact
    # same manifest route; the browser overlays them as ghosts. Loading the
    # manifest instead of drawing another random goal makes the picture auditable
    # against the numbers that selected best.pt.
    config["checkpoint"] = {
        "run": path.parent.name,
        "file": path.name,
        "step": int(checkpoint.get("step", 0)),
        "benchmarkSuccess": checkpoint.get("benchmarkSuccess"),
        "benchmarkSeconds": checkpoint.get("benchmarkSeconds"),
        "loadedAt": time.time(),
    }

    race = None
    if (shape.get("world") or {}).get("goals"):
        manifest_path = path.parent / "benchmark.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"movement run {path.parent.name} has no fixed routes yet (benchmark.json): "
                "they are written by its first exam, so try again once that has finished"
            )
        manifest = json.loads(manifest_path.read_text())
        scenarios = manifest.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise RuntimeError(f"{manifest_path} has no fixed scenarios")
        config = fixed_movement_world(
            config,
            shape,
            levels=config["levelFiles"],
            generated_maps=int(config.get("levelPool", 0)),
            envs=1,
            seed=int(manifest.get("seed", scenarios[0]["seed"])),
            episode_ticks=int(manifest.get("episodeTicks", shape.get("episodeTicks", 1800))),
        )
        config.update(
            agents=agents,
            speed=args.speed,
            port=args.port,
            host=args.host,
            publicOrigin=args.public_origin or None,
            basePath=args.base_path or "",
            race={
                "schemaVersion": manifest.get("schemaVersion", 1),
                "fingerprint": manifest.get("fingerprint"),
                "episodeTicks": manifest.get("episodeTicks", shape.get("episodeTicks", 1800)),
                "scenarios": scenarios,
                # The first ghost is the exact deterministic policy used to
                # select best.pt. The others sample the same policy, so the
                # screen shows both its score-producing path and its spread.
                "racerModes": (
                    ["benchmark"] * agents
                    if args.greedy
                    else ["benchmark", *(["sample"] * max(0, agents - 1))]
                ),
            },
        )
        race = config["race"]

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
        saved_reward = checkpoint.get("reward")
        print(
            f"watching {path.parent.name} ({checkpoint.get('step', 0):,} steps"
            + (
                f", reward {saved_reward:.2f}"
                if isinstance(saved_reward, (int, float))
                else ""
            )
            + (
                f", {agents} isolated ghosts over {len(race['scenarios'])} fixed routes"
                if race else ""
            )
            + f") — {layout['viewer']}",
            flush=True,
        )
        vector_bytes = layout["agents"] * layout["vectorSize"] * 4
        patch_bytes = layout["agents"] * layout["patchCells"]
        use_patch = shape.get("usePatch", True) and patch_bytes > 0
        map_bytes = layout["agents"] * layout.get("mapCells", 0)
        restart_bytes = layout.get("restartBytes", 0)
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
            restarts = None
            if restart_bytes:
                restarts = torch.from_numpy(
                    np.frombuffer(
                        frame,
                        dtype=np.uint8,
                        count=restart_bytes,
                        offset=vector_bytes + patch_bytes + map_bytes,
                    ).copy()
                ).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                if args.greedy:
                    logits, _, carried = policy(
                        vectors, patches, maps, carried, restart=restarts
                    )
                    heads = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
                elif race:
                    # The fixed benchmark that selected best.pt is greedy.
                    # Keep one visible racer on that exact rule while the
                    # remaining ghosts show stochastic attempts from the same
                    # weights. Sampling every ghost made Watch look worse than
                    # the score it claimed to replay.
                    logits, _, carried = policy(
                        vectors, patches, maps, carried, restart=restarts
                    )
                    heads = torch.stack(
                        [Categorical(logits=head).sample() for head in logits],
                        dim=1,
                    )
                    heads[0] = torch.stack([head[0].argmax() for head in logits])
                else:
                    heads, _, _, _, carried = policy.act(
                        vectors,
                        patches,
                        maps,
                        want_entropy=False,
                        carried=carried,
                        restart=restarts,
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
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    main()
