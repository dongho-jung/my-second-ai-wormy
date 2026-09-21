"""Put a trained policy into a real room on webliero.com.

Loads a checkpoint and drives `src/live/drive.js`, which opens the game windows,
gets them into one room and presses the keys. The physics are the same file the
policy trained against — the checksum says so — and `src/live/keys.js` proves
every action it can take is one a player could press. What is left is the two
things a live game has and a headless one does not: a network, and a keyboard
sampled sixty times a second.

    npm run play                      # three worms, the newest run's best policy
    npm run play -- --room-url URL    # join a room that already exists

WebLiero asks for a CAPTCHA to create a room. It is asked for in the game window
and waited out there; nothing here tries to get past it.
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
from watch import newest_checkpoint
from workers import REPO, _read_frame

DRIVER = REPO / "src" / "live" / "drive.js"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", type=int, default=3, help="worms in the room, all driven by the policy")
    parser.add_argument("--run", default=None, help="run id to play; the newest by default")
    parser.add_argument("--checkpoint", default=None, help="a .pt file, overriding --run")
    parser.add_argument("--room-url", default=None, help="join this room instead of making one")
    parser.add_argument("--room-name", default="BOT TRAINING ROOM")
    parser.add_argument("--public", action="store_true", default=True,
                        help="list the room publicly, so passers-by join and are learned from")
    parser.add_argument("--private", dest="public", action="store_false")
    parser.add_argument("--greeting", default="",
                        help="lines to say when somebody joins, separated by |. Empty says "
                             "nothing, which is the default: a room does not need a bot "
                             "introducing itself")
    parser.add_argument("--farewell", default="G G",
                        help="lines to say when a match ends, separated by |")
    parser.add_argument("--yield-to", type=int, default=0,
                        help="give the seats up once this many people are playing. 0 never "
                             "yields; 2 keeps a quiet room company and leaves a busy one alone")
    parser.add_argument("--names", default="BOT FOO,BOT BAR,BOT BAZ",
                        help="one name per driven worm, in order")
    parser.add_argument("--colours", default="220,60,50 70,200,90 70,120,230",
                        help="one r,g,b per driven worm, in order")
    parser.add_argument("--room-size", type=int, default=20,
                        help="seats in the room; it is private, so leave room for people to join")
    parser.add_argument("--nickname", default="Wormy")
    parser.add_argument("--decide-hz", type=float, default=15.0,
                        help="decisions a second; 15 is the 4-tick frameskip it trained on")
    parser.add_argument("--map-ms", type=int, default=1000,
                        help="how often the whole level is re-read; they dig through it as they play")
    parser.add_argument("--cdp-port", type=int, default=9334)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--greedy", action="store_true", help="take the likeliest action instead of sampling")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="three worms at 15 Hz is far too small to be worth a GPU")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--fresh", action="store_true",
                        help="open new tabs and a new room instead of taking over the ones already seated")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def compatible(shape, wants) -> bool:
    """Whether a checkpoint was trained on the observation this room gives."""
    return (
        shape.get("vectorSize") == wants["vectorSize"]
        and list(shape.get("headSizes") or []) == list(wants["headSizes"])
        and list(shape.get("patchShape") or []) == list(wants["patchShape"] or [])
        and bool(shape.get("usePatch", True)) == wants["usePatch"]
        and bool(shape.get("useMap", False)) == wants["useMap"]
    )


def best_available(runs: Path, wants):
    """The most-trained checkpoint that still fits this room, and its shape.

    `best.pt` is each run's own high-water mark, kept on smoothed episode
    reward, so it is the one to take from whichever run is furthest along.
    Reward cannot be compared across runs — the weights and the opponent both
    change — so "furthest along" is decided on steps, among the checkpoints
    that match the observation. Matching is what stops a run from before an
    observation changed being picked merely for being large.
    """
    best = None
    for run in sorted(runs.glob("*/"), reverse=True):
        for name in ("best.pt", "policy.pt"):
            path = run / name
            if not path.exists():
                continue
            try:
                carried = torch.load(path, map_location="cpu", weights_only=False)
            except (OSError, RuntimeError, EOFError):
                continue  # still being written
            if not compatible(carried.get("layout", {}), wants):
                continue
            step = int(carried.get("step", 0))
            if best is None or step > best[2]:
                best = (path, carried, step)
            break
    return best


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
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    shape = checkpoint["layout"]
    trained_agents = shape.get("agents", 3)
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

    config = {
        "players": args.players,
        # The vector keeps the width the policy was trained on, however many
        # worms are in the room.
        "observationFoes": trained_agents - 1,
        "roomUrl": args.room_url,
        "roomName": args.room_name,
        "roomSize": args.room_size,
        "nickname": args.nickname,
        "nicknames": [part.strip() for part in args.names.split(",") if part.strip()],
        "isPublic": args.public,
        # Play while the room is short of people and spectate once it is not:
        # a seat held by a bot is a seat somebody else cannot have.
        "yieldTo": args.yield_to,
        "greeting": args.greeting,
        "farewell": args.farewell,
        "colours": [
            [int(channel) for channel in triple.split(",")]
            for triple in args.colours.split()
        ],
        "decideHz": args.decide_hz,
        "mapMs": args.map_ms,
        "cdpPort": args.cdp_port,
        "frameskip": shape.get("frameskip", 4),
        "fresh": args.fresh,
        "verbose": args.verbose,
    }
    if args.profile:
        config["profile"] = args.profile

    driver = subprocess.Popen(
        ["node", str(DRIVER), json.dumps(config)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(REPO),
        bufsize=0,
    )
    try:
        layout = json.loads(_read_frame(driver.stdout))
        if layout["vectorSize"] != shape["vectorSize"]:
            raise RuntimeError(
                f"the policy wants a vector of {shape['vectorSize']} and the game "
                f"gives {layout['vectorSize']}"
            )
        # The vector is not the whole observation, and a patch of the wrong size
        # reshapes into nonsense rather than failing.
        wanted_cells = shape.get("patchShape") and shape["patchShape"][-2] * shape["patchShape"][-1]
        if shape.get("usePatch", True) and wanted_cells and layout["patchCells"] != wanted_cells:
            raise RuntimeError(
                f"the policy looked at {wanted_cells:,} patch cells and the game "
                f"gives {layout['patchCells']:,}"
            )
        print(
            f"playing {path.parent.name} ({checkpoint.get('step', 0):,} steps"
            + (f", reward {checkpoint['reward']:.2f}" if "reward" in checkpoint else "")
            + f") in {layout['room']}",
            flush=True,
        )
        vector_bytes = layout["agents"] * layout["vectorSize"] * 4
        patch_bytes = layout["agents"] * layout["patchCells"]
        use_patch = shape.get("usePatch", True) and patch_bytes > 0
        map_bytes = layout["agents"] * layout.get("mapCells", 0)
        use_map = shape.get("useMap", False) and map_bytes > 0
        heads_count = len(layout["heads"])
        wants = {
            "vectorSize": shape["vectorSize"],
            "headSizes": shape["headSizes"],
            "patchShape": list(shape.get("patchShape") or []),
            "usePatch": shape.get("usePatch", True),
            "useMap": shape.get("useMap", False),
        }
        # A policy loaded once is the policy the room keeps for as long as it is
        # open, however far the training has moved on meanwhile — which is how
        # a room spent an evening showing a 172,800-step checkpoint while the
        # run behind it passed 2.6M. So it is picked up again whenever a round
        # ends. The driver zeroes the vector of every worm it cannot see, so an
        # all-zero frame is nobody alive: between rounds, or everyone dead.
        playing_step = int(checkpoint.get("step", 0))
        anyone_alive = True
        looked_at = 0.0
        # Carried from one decision to the next: without this the policy is
        # handed a blank memory every frame and can hold nothing at all.
        carried = None
        last_refusal = None
        while True:
            frame = _read_frame(driver.stdout)
            vectors = torch.from_numpy(
                np.frombuffer(frame, dtype=np.float32, count=vector_bytes // 4)
                .reshape(layout["agents"], -1)
                .copy()
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
            was_alive, anyone_alive = anyone_alive, bool(vectors.any())
            if was_alive and not anyone_alive and not args.checkpoint:
                now = time.monotonic()
                # Worms die often; the directory is only re-read on a round that
                # ends at least this long after the last look.
                if now - looked_at > 20:
                    looked_at = now
                    found = best_available(runs, wants)
                    if found and found[2] > playing_step:
                        # Not `carried`: that name holds the policy's memory in
                        # this loop, and reusing it here put a checkpoint dict
                        # where the memory goes on the very next decision.
                        path, newer, step_of = found
                        try:
                            policy.load_state_dict(newer["policy"])
                        except RuntimeError as error:
                            # A checkpoint from a different network shape. Keep
                            # playing what we have rather than falling over.
                            log_once = f"cannot use {path.parent.name}: {error}"
                            if log_once != last_refusal:
                                print(log_once.split("\n")[0], flush=True)
                                last_refusal = log_once
                        else:
                            playing_step = step_of
                            policy.eval()
                            carried = None  # a different policy, a fresh memory
                            print(
                                f"now playing {path.parent.name} ({playing_step:,} steps"
                                + (f", reward {newer['reward']:.2f}" if "reward" in newer else "")
                                + ")",
                                flush=True,
                            )
            with torch.no_grad():
                if args.greedy:
                    logits, _, carried = policy(vectors, patches, maps, carried)
                    heads = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
                else:
                    heads, _, _, _, carried = policy.act(
                        vectors, patches, maps, want_entropy=False, carried=carried
                    )
            block = heads.to(torch.uint8).cpu().numpy().tobytes()
            driver.stdin.write(struct.pack("<I", heads_count * layout["agents"]) + block)
            driver.stdin.flush()
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    finally:
        try:
            driver.stdin.close()
        except OSError:
            pass
        try:
            driver.wait(timeout=10)
        except subprocess.TimeoutExpired:
            driver.kill()


if __name__ == "__main__":
    main()
