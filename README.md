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
| `actions.js` | The key bitmask (`1 left … 256 dig`), the rope and weapon-change messages that are not bits, the eight heads a policy emits, and the hold that keeps a rope throw from being undone for a while. Exactly how a real room passes input: there is no "release" head, because letting go of the rope is a press of Jump. |
| `view.js` | Turns a headless world and a live snapshot into **one shape**. The observation encoders read only that. |
| `observation.js` | The vector (150-238 numbers, depending on how many worms are playing), the worm's own 213x121 view of the terrain in eight planes — rock, dirt, free, the flagless ground a rope goes through, shots, foes, itself, its destination — and the whole level on a 32x32 grid. Any of them can be left out. |
| `progress.js` | Whether a worm is stuck, going in circles, or closing on a goal, and how long it has had the goal. |
| `reward.js` | Damage dealt minus damage taken, kills and deaths, the movement terms, the ladder up to aiming, and a price per rope throw when a run asks for one. Blowing yourself up costs the same as being shot. |
| `env.js` | `reset()` / `step()`, frameskip, respawn, input latency, the goals of the movement task and how far away they are drawn. An episode ends on a clock, so its last observation is handed over to be valued rather than thrown away. Letting go of the rope is a jump press, as it is in the game. |
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

A death nobody else caused — its own rocket, a fall — is counted apart, and
`--suicide-cost` can charge for it on top of the death. It is off by default:
a kill pays 4 and a death costs 2, so a worm that blows itself up on the way to
one kill still comes out ahead, and whether charging for that makes a better
player or a shyer one is for `npm run evaluate` to say.

The weights are `DEFAULT_WEIGHTS` in `src/env/reward.js`, and the reasoning
behind each one — including the run that learned to stand still — is in the
comments beside them. `docs/network.html` draws the whole policy in Korean.

### Learning to move first

```bash
npm run train -- --task movement --agents 6 --rope-throw-cost 0.01
```

`--task movement` turns the fighting terms off, holds the trigger shut, and
hands every worm one place on the map to reach. Once every worm either arrives
or uses its `--goal-patience`, a fresh episode begins with new random tasks.
Training stays varied so the policy cannot memorise a route, but one lucky
sequence of short destinations cannot collect repeated arrival rewards.

Before the first update and every 20 updates after that, the current policy
takes a separate exam: 34 isolated one-worm scenarios with a fixed seed, map,
spawn, destination and 30-second horizon. Actions are greedy, so repeated
checks do not add sampling luck. The default suite visits each of the 17 room
maps twice instead of leaving map coverage to a random draw. `best.pt` is
selected lexicographically by fixed-scenario successes, obstructed-scenario
successes, then failure-aware time, speed and path efficiency.
The full task list is written to `benchmark.json`; it is never used for
gradients.

The destinations start close. `--goal-radius 96-1600` draws them within 96 px
of the worm at first — a walk or a jump — and moves the limit out to 1,600 px,
so the rope is met when walking stops being enough rather than on the first
decision. What moves it is `--goal-curriculum`: `steps` grows it with the clock
over the first `--goal-grow` of the run, and `success` grows it a tenth when
85% of a world's last `--goal-window` destinations were reached and brings it
back a tenth when half or fewer were. The clock outran the first runs — six
destinations a match at 130 px, three by the time it had moved out to 250 —
which is what the second mode is for. The first runs drew from the
whole map, 450-550 px away on average, and a policy that could not walk yet
learned the one thing that covers that distance: re-throwing the rope along
its aim several times a second, which the engine reels it in on. It reached
six destinations a minute that way and never learned to hold a rope. The rope
in this mod is a grappling hook — it flies at 8 px a tick, holds at a fixed
28 px of length and pulls the worm toward the anchor on its own — so
`--rope-throw-cost` charges a little per throw to make holding on the cheaper
way to cover the same ground. Both are measured in
`docs/movement-runs-2026-09-22.md`.

### Fine-tuning a movement policy for speed

Once the success curriculum has reached its far radius, more of the same
reward mostly teaches reliability. Per-pixel progress telescopes to nearly the
same total whether a route takes three seconds or twenty. Carry the best
checkpoint into a speed phase instead of starting the movement lesson again:

```bash
npm run train -- \
  --task movement \
  --resume artifacts/runs/<parent>/best.pt \
  --total-steps <additional-speed-phase-steps> \
  --goal-radius 1600 \
  --goal-above 0.5 \
  --goal-detour 0.5 \
  --goal-progress-mode best \
  --rope-hold 12 \
  --goal-patience 450 \
  --goals-per-episode 1 \
  --goal-arrival-reward 8 \
  --goal-speed-reward 1 \
  --goal-speed-cap 16 \
  --goal-progress-decay 0.25 \
  --goal-progress-floor 0.1 \
  --keep-every 100 \
  --label "speed fine-tune"
```

`--total-steps` is the number of decisions to add after the checkpoint; the
saved counter continues from the checkpoint's step. The speed bonus is paid
only on arrival and uses direct pixels per decision, capped before its weight
is applied. The old per-pixel shaping fades over the requested share of this
speed phase, while the arrival and speed rewards stay. A fixed 450-decision
deadline gives every A/B run the same 30-second limit. `--goal-patience-min`
can still enable an adaptive training curriculum, while the fixed benchmark
remains the checkpoint selector.

`--goal-detour 0.5` deliberately puts solid terrain across the direct line in
half the tasks. These include ledges where the first useful move is sideways
or away from the destination. `--goal-progress-mode best` pays only when the
worm sets a new closest distance: taking the necessary step away is neutral,
then rounding the obstruction and getting closer pays again. Returning over
already credited ground pays nothing, so the policy cannot score by pacing.
The fixed benchmark reports its obstructed subset separately as detour success
and detour time, so an easy straight route cannot hide this failure mode.

The monitor leads with fixed-exam success and failure-aware seconds. It also
reports the random training tasks' assigned distance, `goalSeconds`, direct
`goalSpeed`, and `goalPathEfficiency`, plus goals reached and missed. Death and
respawn time remains on the goal clock, and the teleport itself is excluded
from route length, so dying cannot masquerade as fast travel.

Compare the result with its parent on fixed conditions before keeping it:

```bash
npm run evaluate -- \
  --left artifacts/runs/<speed>/best.pt \
  --right artifacts/runs/<parent>/best.pt \
  --task movement \
  --goal-radius 1600 \
  --goal-patience 450 \
  --episodes 96
```

Movement evaluation runs the policies separately on the exact same fixed map,
spawn and destination list. Each scenario has one worm, one goal and one
deadline, so neither terrain interference nor a lucky goal sequence can decide
the result. Success is primary; failures cost the full horizon, then time,
direct speed and path efficiency explain the difference. Mutating whole neural
networks and keeping the lucky few would spend far more simulations
rediscovering a policy that PPO can already improve from the working checkpoint.

Five of the community maps draw much of their walls in a colour that has no
material flags: a worm walks into it and a rope flies through it. The patch
and the map show that ground as its own kind, `ghost`, so the policy can see
which walls hold a rope, and both show where the goal is when it is in view.

**A throw is a commitment.** `--rope-hold 12` keeps a rope for twelve
decisions after it is thrown — another throw is ignored and the jump key,
which is how a rope is let go, is dropped — so the rope gets to pull. Without
it a trained policy kept a rope for two decisions in the median and was pulled
nowhere: it had a "release" choice of its own that also pressed Jump (it had
to, to match the game), used it as a second jump key 275 times a match, and
undid every throw with it. There is no release choice any more; the jump head
is the Jump key, for the policy as for a person, and the live driver keeps the
same hold. Two more things make the rope a mechanism rather than a lottery:
the vector says what a throw would hook right now — along the aim, how far to
the first ground that holds a rope and how far up it is — and `--goal-above
0.5` draws half the destinations where a jump does not reach, so the success
curriculum cannot move on until the rope is used.

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
- **Every fifty updates the policy is probed on its own.** It plays a few
  whole matches against worms that press nothing, and the page charts what
  happened as `probeKills`, `probeDeaths`, `probeSuicides` and the damage both
  ways. Everything else on the page is measured against a moving target, the
  policy itself or its recent past; this is the one figure that says whether it
  can find and kill a worm that just stands there, and how often it kills
  itself trying. `--probe-every` sets the interval, 0 turns it off.
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
  longer than one frame has to be carried. 1.86M parameters at two pixels a
  patch cell, 1.26M at four.
- **The entropy of every head is logged**, not only their sum. Eight heads
  summed into one number cannot tell a rope head that has gone deterministic
  from a fire head the bonus keeps uniform, and in a movement run that is the
  question. The log line carries the six that move a worm.
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
- **The mod's weapons have to be measured first.** `npm run weapons` fires every
  weapon in a controlled world and writes `artifacts/weapons.<mod>.json`: how
  fast a shot goes, how far it drops, what it does to a target and to whoever
  fired it. That file is where the policy's weapon features and the aim reward's
  ballistics come from. Without it the engine has neither — 72 fields of the
  vector stay zero and the aim rewards pay nothing — and every chart still looks
  healthy, which is how the first cluster run spent two and a half hours blind.
  So the trainer now refuses to start without it, the file is committed, and the
  image build copies it in. If the mod changes, the engine refuses the stale
  file too; measure again.
- PyTorch lives in `artifacts/.venv` (gitignored). `npm run train` says how to
  make it if it is not there.

Checkpoints and metrics land together in `artifacts/runs/<id>/`. The best policy
so far is kept separately from the latest one, because self-play wanders. "Best"
is the learners' combat score — damage, kills and deaths, with none of the ladder
in it — because the ladder fades over a run, and on the total reward a later,
better policy reads lower than an earlier one.

## Watching a policy play

```bash
npm run watch                        # the newest run's best policy
npm run watch -- --agents 5 --speed 2
```

There is also a **Watch** button on the training page, which starts this for the
selected run and opens it. For a movement run it replays `benchmark.json`: all
coloured worms get the exact same fixed map, start, goal and clock. Each ghost
owns a separate engine world, including its own worms, projectiles and mutable
terrain, so collisions, ropes, shots and digging cannot change another
attempt. Ghost 1 uses the greedy actions that produced the fixed benchmark
score; the remaining ghosts sample the same policy to show its variation.
Finish rank and time are shown before the next fixed route starts.

For a combat run, one match is played at the speed the game actually runs at
and drawn in the browser on port 8769: the terrain the worms are digging
through, where they are aiming, what they are holding, and the score.

This is **the headless engine rendered, not a room on webliero.com**. The
physics are identical — the same bundle, checked by the same checksum.

## Playing in a real room

```bash
npm run play                              # three worms, the newest run's best policy
npm run play -- --room-url URL --players 2
npm run record -- --patch-scale 4         # write down what the people in the room do
```

`npm run play` loads a checkpoint and drives `src/live/drive.js`, which opens
the game windows, gets them into one room and presses the keys the policy
chooses (`src/live/keys.js` proves every action it can take is one a player
could press). WebLiero asks for a CAPTCHA to create a room; it is waited out in
the window, never worked around.

The policy has to meet **the observation it trained on**, or it is playing a
different game on its first real match. The driver therefore builds the vector
with the mod's measured weapon profile, cuts the patch at the checkpoint's own
scale, and refuses a room running another mod. `npm run record` needs the same
care: pass `--patch-scale` to match the run that will learn from the recording,
because a recording at the wrong scale is skipped, not used.

## Comparing two policies

```bash
npm run evaluate -- --left artifacts/runs/<a>/best.pt --right artifacts/runs/<b>/best.pt
npm run evaluate -- --left artifacts/runs/<a> --right random      # against keys pressed at random
npm run evaluate -- --left artifacts/runs/<a> --right still       # against a worm that does nothing
npm run evaluate -- --history artifacts/runs/<a>                  # against its own earlier selves
npm run evaluate -- --left latest --right still                   # the newest run, against still
```

Nothing on the training page can compare two runs: every figure there is a
policy measured against itself or its own recent past, and a run that improves
slowly and one that improves quickly can show the same **is it beating its past
self**. Combat evaluation seats the checkpoints in the same free-for-all and
swaps sides on matched maps. Movement evaluation instead isolates each policy
and gives both the exact same fixed map, spawn and one-goal scenarios. It
reports success, failure-aware time, speed and route efficiency with a bootstrap
interval on the paired difference. Movement comparisons use the far end of the
checkpoint's goal radius by default and freeze both curricula; pass
`--goal-radius` and `--goal-patience` to set the held-out conditions explicitly.
`random` and `still` are two bars that never move, so a policy
can be measured against the same thing early in a run and late in it. Both
sides must have been trained on the same vector; two patch scales can share a
match, because the world cuts the same ground twice and shows each side the
cut it learned on.

A run trained with `--keep-every N` also keeps a `policy-<steps>.pt` every N
updates, and `--history` seats its best against each of them in turn: the
progress of a run measured on the field, which the reward curve cannot give
because the reward itself is what changes. The table it writes beside the
checkpoints shows up on the training page as **Progress on the field**.

## Running it somewhere else

The learning half goes in a container. The watching half does not — it drives a
real browser and has nothing to do in a cluster — so `playwright`, the one npm
dependency this project has, is never installed and nothing under `src/env/`,
`train/` or `scripts/train.js` imports anything outside Node's own builtins.

```bash
docker build -t wormy .
docker run --rm wormy node scripts/train.js --total-steps 50000
```

**Most of `artifacts/` is not in the repository, and the build copies in only
what is.** The game, its mod, the maps the room plays and the measured weapon
profile are committed, so a build needs no network and every build gets the
same bytes; runs, recordings and the venv are not. The three fetch scripts run
during the build and check what is there — the bundle's SHA-256 against the one
`src/adapter-v20.js` was read off, so a build against a moved version fails
rather than training on a field mapping that no longer means what it says — and
fetch whatever is missing. A fresh clone can run them too:

```bash
npm run engine     # the four files the headless engine is
npm run mods       # the game the room runs
npm run maps       # the maps it runs them on
npm run weapons    # measure the mod's weapons; training refuses to start without this
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
train/       ppo.py · policy.py · evaluate.py  (PyTorch, MPS)
  ↓
artifacts/runs/<id>/  metrics.jsonl · benchmark.json · best.pt · policy.pt
  ↓ src/train/monitor.js            ↓ src/env/watch.js
public/train/  training page (8768)   public/watch/  match viewer (8769)
```

The upstream game's source, its network responses and its key handlers are never
touched, and no observation globals are added to the browser. The field mappings
are not an official API, so an upstream change means checking the adapter again.
