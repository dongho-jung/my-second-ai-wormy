# my-second-ai-wormy

Reads [WebLiero](https://www.webliero.com/) state and terrain live and draws it
on a local dashboard — and runs the same game engine headless, thousands of
times faster than real time, to train a policy to play it.

The watching half is **state collection, terrain extraction and drawing**, taken
out of `../my-first-ai-wormy`. The model calls, the reflex loop, key input and
control APIs, telemetry, self-play and the self-improvement loop from that
repository are all deliberately absent here: **against a running game this tool
only looks, and has no control surface at all.**

The learning half never touches a running game. It loads the same official
bundle into plain Node, plays matches at a few thousand times real time, and
trains on them.

## Running

```bash
npm start
```

The game window and the dashboard open as **two separate windows, each taking
exactly half the screen** (game on the left, dashboard on the right). State and
terrain start being drawn **once you join a room in the game window**. A CAPTCHA,
if one is asked for, has to be solved there. Ctrl+C, or closing the game window,
stops.

Node **26** is required and the version is pinned in `.nvmrc`.

```bash
npm start -- --help   # every option
npm test              # adapter, server, environment and monitor tests, no game needed
npm run preview       # the dashboard alone, drawn from the test fixture
npm run rollout       # episodes in the headless environment, random policy
npm run train         # self-play PPO
npm run watch         # watch a trained policy play
npm run monitor       # the training page (8768)
```

### Browser and profile

- Chromium is left on its debugging port (9334 by default) and the next run
  **attaches** to it, so the room and an already-solved CAPTCHA survive a
  restart. `--own-browser` makes it exit with the process instead.
- One profile at `~/.cache/wormy-ii/chrome-profile` is **reused every time**, so
  nickname, key bindings and the rest of the game's settings survive a browser
  restart. `--profile PATH` moves it or keeps more than one.
- **Tab restore data (`Default/Sessions`) is deleted on every launch.** Without
  that, the previous run's tabs come back and pile up in one window. Settings,
  cookies and localStorage are untouched.
- Every launch clears out what the last one left behind — blank tabs, dead
  dashboards, duplicate game tabs — back to **two windows, two tabs**.
- Ports: dashboard 8766, debugging 9334, fixture preview 8767, training monitor
  8768, match viewer 8769. None of them collide with `../my-first-ai-wormy`
  (8765 / 9333), so both can run at once without touching each other.

## What is extracted

`snapshotV20` in `src/adapter-v20.js` runs inside the page over CDP and returns
one of three things.

| Request                  | Contents                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| (default)                | tick, room and match, map name and size, players and worms (position, velocity, health, input bitmask, aim, weapons, rope), projectiles, pickups, flag |
| `{ terrain: true }`      | every pixel of the level (one palette-index byte each) plus 768 bytes of palette and 256 material flags          |
| `{ terrainPatch: true }` | the **426x240 window** around the worm, one byte per pixel, plus contacts, walk probes and ceiling/floor/wall distances |

Terrain is read as **material flags, not palette indices**. Bit 3 is the
background a worm may stand in, bits 0-1 the dirt a weapon digs through, bit 2
the rock it cannot. Levels each pick their own colours, so an index range says
nothing.

Version lock: collection only starts when the SHA-256 of the official
`/v/20/game-min.js` matches. If the bundle changes it reports
`unsupported_client` rather than guessing with the old mapping. **Do not just
swap the checksum.**

## What is drawn

- **Map** — the whole level at **one pixel per pixel**, in the game's own
  palette, with a `material` checkbox that switches to a material view (air,
  dirt, rock). Worms, projectiles, pickups and ropes are drawn over it, and the
  Near window's position is outlined.
- **Near** — the 426x240 window the game draws around the worm, also **1:1**.
  Outside the map is shaded so it is not mistaken for air.
- **Surroundings** — whether a step is possible (clear, step, dirt, rock), the
  engine's own contact counts and automatic step-up, and the pixel distances to
  the ceiling, floor and side walls.
- **Worm / Players** — health, position, velocity, aim, weapons and ammo, rope,
  the keys currently held, and every player's score and state.

State refreshes at 20 Hz over SSE, the surroundings every 250 ms, the whole
level every 3 s. Because the three run on different clocks, each picture carries
**when it was read**.

## Local HTTP (read-only)

| Endpoint       | Contents                                             |
| -------------- | ---------------------------------------------------- |
| `GET /state`   | the latest state                                     |
| `GET /events`  | the same state over SSE (32 clients maximum)         |
| `GET /terrain` | the window around the worm and the measured surroundings |
| `GET /map`     | every level pixel, the palette and the material flags |
| `GET /health`  | whether the service is up (`ok`) and has state (`ready`) |

It binds to 127.0.0.1 only and refuses another Host or Origin, and any method
but GET. If samples stop arriving, `/state` answers `stale` with `game: null`.

## The headless training environment

`src/env/` runs the official v20 bundle **in plain Node with no browser, no
server and no rendering**. It exists to train on, and shares nothing with the
dashboard half but the adapter's checksum.

```bash
npm run rollout -- --episodes 40 --level-pool 8
npm run rollout -- --help
```

The default match is **three worms in a free-for-all**, but the count is one
setting: solo, a duel and a five-way brawl all run on the same code.

| Module | What it does |
| --- | --- |
| `engine.js` | Patches the bundle in memory, evaluates it, and checks its SHA-256 against the one the adapter locked. Also seed-reproducible level generation, random weapon loadouts, and the instrument that records **who hit whom**. |
| `actions.js` | The key bitmask (`1 left … 256 dig`), the rope and weapon-change messages that are not bits, and the eight heads a policy emits. Exactly how a real room passes input. |
| `view.js` | Turns a headless world and a live snapshot into **one shape**. The observation encoders read only that. |
| `observation.js` | The vector (132-192 numbers, depending on how many worms are playing), the worm's own 213x121 view of the terrain, and the whole level on a 32x32 grid. Any of them can be left out. |
| `progress.js` | Whether a worm is stuck, going in circles, or closing on a goal. |
| `reward.js` | Damage dealt minus damage taken, kills and deaths, the movement terms, and the ladder up to aiming. Blowing yourself up costs the same as being shot. |
| `env.js` | `reset()` / `step()`, frameskip, respawn, input latency. An episode ends on a clock, so its last observation is handed over to be valued rather than thrown away. |
| `vec.js`, `worker.js` | Many worlds in one process, and the binary frames the trainer talks over. |

- **The bundle and its assets are not in the repository** (`artifacts/` is
  gitignored). Without them the environment tests skip with a reason. Fetch them
  with `artifacts/headless-sim/fetch-assets.sh`.
- That the physics match the live game rests on **one checksum**. If the bundle
  changes, loading refuses.
- Measured (M2 Pro, one core, two agents): vector only **76,485 steps/s
  (2,549× real time)**, with the terrain patch **36,509 steps/s (1,217×)**.
- The same seed and the same actions give the same episode. Tests compare
  positions and health to six decimals.

### What is rewarded

Damage dealt and kills **add**; damage taken and deaths **subtract**. With three
or more worms, "the other one lost health" is not a signal — two of them can be
fighting while the third watches — so the attacker's id is taken **from the
engine's own damage function** and credited exactly. Damage from your own
explosion or a fall arrives with your own id, so it is taken without being
dealt.

On top of that, **being stuck or going in circles costs**. Failing to move 14
pixels in four seconds is stuck; coming back to a cell you were recently in is a
circle. They are different failures and are measured differently. Ground you
have never covered pays, as a **share of the map** rather than a count of cells,
and only the first time: a route walked twice earns nothing the second time. Set
a goal position and closing on it pays.

Aiming pays too, in small amounts, because it has to. Nothing in this game
rewards pointing a gun — the payoff arrives later as damage, if the shot lands —
so a policy that cannot aim never fires well enough to discover that aiming was
the point. Pointing at somebody you could actually hit pays per decision, and
firing while lined up pays more. These are the ladder, not the destination, and
they do not anneal on their own.

The weights are `DEFAULT_WEIGHTS` in `src/env/reward.js`, and the reasoning
behind each one — including the run that learned to stand still — is in the
comments beside them. `docs/network.html` draws the whole policy in Korean.

## Training

```bash
npm run train -- --agents 6 --opponents 0.34 --total-steps 20000000
npm run train -- --help
```

- **Six worms a match is the better bet, measured.** Against three, at the same
  number of worms in flight, it costs about five per cent of the throughput and
  pays back **more than three times the kills and four times the damage**: a
  three-way free-for-all spends most of its early life with nothing happening,
  and the reward is made of things happening. The buffers are
  `steps × workers × envs × worms`, so doubling the worms means halving `--envs`
  to stay in the same memory. `compose.yaml` and `deploy/train-job.yaml` both
  run six; the code's own default is still three.
- **Every worm shares one policy**, unless `--opponents` says otherwise. In a
  free-for-all that is self-play by construction: whatever one of them learns
  it immediately has to face, and there is no opponent to hand-write. It is also
  why the average reward going up is not the same as getting better — all three
  getting more reckless together looks identical. `--opponents 0.34` hands one
  worm in three to an older copy of the policy — two of six — and learns from
  nothing they do, so the number it is scored against stays still while the
  policy moves. Each opponent seat gets its own generation, so a match with two
  of them is two different past selves rather than one standing in two places.
  The training page's first headline, **is it beating its past self**, is that
  difference; without opponents it has nothing to compare and says so.
- **The ladder can be taken away.** Aiming, closing in and covering ground pay
  because nothing else would get a policy started, and each is also a way to
  score without playing well. `--shaping-decay 0.6` fades them over the first
  three fifths of a run and leaves the rest on damage, kills and deaths. Off by
  default, because no run has yet been long enough to say what it should be.
- The engine is JavaScript and the training is **PyTorch on Apple MPS**. They
  meet on a worker process's own stdin and stdout in binary frames — no port to
  pick, no socket to clean up, and the workers die with the parent.
- The observation is **mixed**, as designed: state in a vector, the terrain
  through two convolutions — the worm's own view, and the whole level small —
  and a **GRU** on top of the joined features, because a decision that takes
  longer than one frame has to be carried. 1.59M parameters.
- **A rollout is 128 decisions and the gradient runs through 32 of them.** The
  two are not the same knob. How far a reward can be from the action that
  earned it and still reach it is the first; how far back the memory learns is
  the second. Both were 12 — eight tenths of a second — and a mine went off in
  the next rollout, after the throw had been learned from and discarded.
- Matches are played on the **community maps the room runs** (`npm run maps`).
  `--maps 12` mixes the game's own generated dirt back in.
- **Measure it before sizing a run.** The last figure recorded here — 6,570
  steps/s on an M2 Pro — was a 0.52M network on 12-step rollouts with no memory,
  and none of those is true any more. A container on one core of this machine
  manages a tenth of that per worm. What a run costs is a property of the node
  it lands on, so start one, read `stepsPerSecond` off the training page, and
  divide.
- PyTorch lives in `artifacts/.venv` (gitignored). `npm run train` says how to
  make it if it is not there.

Checkpoints and metrics land together in `artifacts/runs/<id>/`. The best policy
so far is kept separately from the latest one, because self-play wanders.

## Watching a policy play

```bash
npm run watch                        # the newest run's best policy
npm run watch -- --agents 5 --speed 2
```

There is also a **Watch** button on the training page, which starts this for the
selected run and opens it.

One match is played at the speed the game actually runs at and drawn in the
browser on port 8769: the terrain the worms are digging through, where they are
aiming, what they are holding, and the score.

This is **the headless engine rendered, not a room on webliero.com**. The
physics are identical — the same bundle, checked by the same checksum — but
putting a policy into a live online room needs the key-input path this
repository does not have yet.

## Running it somewhere else

The learning half goes in a container. The watching half does not — it drives a
real browser and has nothing to do in a cluster — so `playwright`, the one npm
dependency this project has, is never installed and nothing under `src/env/`,
`train/` or `scripts/train.js` imports anything outside Node's own builtins.

```bash
docker build -t wormy .
docker run --rm wormy node scripts/train.js --total-steps 50000
```

**`artifacts/` is not in the repository and not in the build context.** The
game, its mod and the maps the room plays are fetched during the build, from
the game's own versioned path and from the community pools, and the bundle's
SHA-256 is checked against the one `src/adapter-v20.js` was read off. A build
against a moved version fails rather than training on a field mapping that no
longer means what it says. The same fetch is a script now, so a fresh clone can
do it too:

```bash
npm run engine     # the four files the headless engine is
npm run mods       # the game the room runs
npm run maps       # the maps it runs them on
```

`.github/workflows/image.yml` builds and pushes to `ghcr.io`. A published
package is **private by default even when the repository is public** — opening
it is one manual step on the package's settings page, and nothing in a workflow
can do it for you.

### Trying it locally first

```bash
docker compose up
```

Then <http://localhost:8768>: the training page, following the run as it goes,
with a **Watch** button that opens the match being played on 8769. It builds
the image the first time and keeps the runs in a volume; `docker compose down
-v` takes both away.

Nothing is configured, because unset is the local case — the pages serve at the
root and answer to localhost. The first checkpoint takes a minute, and the
Watch button says so until there is one.

### The two pages, from somewhere that is not this machine

Both bind loopback and refuse a request whose `Host` is not their own, which is
right on a laptop and useless behind an ingress. Three settings move them,
each with an environment variable of its own:

| | | |
| --- | --- | --- |
| `--host` | `WORMY_HOST` | what to bind; `0.0.0.0` in the image |
| `--public-origin` | `WORMY_PUBLIC_ORIGIN` | where a browser actually reaches it |
| `--base-path` | `WORMY_BASE_PATH` | a path it is mounted under, like `/ai-worm` |

The host check stays: it is what stops a page on another site driving this one
through a browser that can reach it. The question it asks is the *name* — an
attacker's domain pointed at a loopback address still arrives carrying its own
name — so loopback answers on whatever port it was published on, and anything
else has to be named through `--public-origin`. The pages ask for their own files by relative
path, so one prefix moves the whole thing, and `/ai-worm` redirects to
`/ai-worm/` so that resolving works.

The training page is 8768. The match viewer the **Watch** button starts is
8769, and it is told the same origin one path along — route `/ai-worm` to the
first and `/ai-worm/watch` to the second, same hostname, and the button opens
something that works. `deploy/train-job.yaml` is an example of the pod; the
Service and the ingress are yours.

## The training monitor

A **separate page on a separate port** from the game dashboard. Training runs
for hours, and there has to be somewhere to see whether it is going well.

```bash
npm run monitor              # http://127.0.0.1:8768
```

- Runs are written to `artifacts/runs/<id>/` as `run.json` and
  `metrics.jsonl`. Append-only text, so a run that is killed halfway is still
  readable.
- The monitor reads only the bytes that have appeared since it last looked and
  streams them to the page. The training side does not need to know anyone is
  watching.
- **Every numeric field becomes a chart**, in the order the run first mentioned
  it, so a trainer that starts logging something new needs no change here. Known
  names get a label and a direction, so rises and falls are green or red.
- It binds to 127.0.0.1 and serves GET. The one exception to being read-only is
  the Watch button, which starts a viewer.

## Structure

```text
game window (official WebLiero, unmodified)
  ↓ Chrome DevTools Protocol (read-only)
observer.js   finds the controller, checks the bundle checksum, calls the adapter
  ├─ stream.js    20 Hz state sampling
  └─ terrain.js   surroundings every 250 ms, whole level every 3 s
       ↓
server.js  /state · /events · /terrain · /map · /health
       ↓
public/    dashboard (map 1:1, near 1:1, numbers)
```

The learning side is entirely separate from the above.

```text
official v20 bundle (same file, same checksum)
  ↓ node vm, no browser
src/env/     engine · actions · view · observation · progress · reward · env
  ↓ vec.js (many worlds) → worker.js (binary frames)
train/       ppo.py · policy.py  (PyTorch, MPS)
  ↓
artifacts/runs/<id>/  metrics.jsonl · best.pt · policy.pt
  ↓ src/train/monitor.js            ↓ src/env/watch.js
public/train/  training page (8768)   public/watch/  match viewer (8769)
```

The upstream game's source, its network responses and its key handlers are never
touched, and no observation globals are added to the browser. The field mappings
are not an official API, so an upstream change means checking the adapter again.
