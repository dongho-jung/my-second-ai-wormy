# Can this learn locally? — findings and plan

Investigated 2026-09-20. The answer first: **yes.** What stood in the way was
never the hardware. It was the assumption that the game only runs at one times
real time, and that assumption is wrong.

Every path in this document is relative to this repository
(`my-second-ai-wormy`). The earlier one is referred to by its absolute path,
`/Users/dongho/projects/my-first-ai-wormy`.

> **This is the state on 2026-09-20 and it is kept as a record, not as the
> current numbers.** The observation, the network and the reward weights have
> all moved since: the patch is the worm's whole 213x121 view rather than
> 32x32, there is a picture of the whole level as well, weapons go in as
> measured behaviour and as a learned identity, the features run through a GRU,
> and the reward is no longer symmetric. For what is true now read
> `src/env/observation.js`, `src/env/reward.js` and `train/policy.py`, which
> carry their reasoning in comments, or `docs/network.html` for the drawing.

## 0. Where this repository stands

**Against a running game it still only looks.** State collection, terrain
extraction and the dashboard have no control API, and that has not changed.

**The offline environment exists** (2026-09-20, stage 1 of section 5).
`src/env/` runs the official v20 bundle with no browser and gives observations,
rewards and determinism. `npm run rollout` plays episodes with a random policy.

**So does the trainer** (stage 3). PPO in `train/` runs self-play — three worms
in a free-for-all by default, and the count is one setting.

```sh
npm run train -- --agents 3 --total-steps 8000000
npm run monitor          # http://127.0.0.1:8768
npm run watch            # play the best policy and watch it
```

Still missing:

- **The path into a live game** — key input. The earlier repository's
  `/Users/dongho/projects/my-first-ai-wormy/src/controls.js` is the reference
  implementation: it writes key bindings into the game's settings and holds each
  pressed key for at least 100 ms so the 60 Hz sampler sees it.

## 1. What was established (all measured on this machine)

### The game engine runs with no browser

In the official v20 bundle the client simulates the whole world.
`World.update()` is one tick and the RNG is a single seeded LCG. With no
browser, no server and no rendering, **it runs as it is in plain Node**. The
bundle's SHA-256 is `b3c7b33c…`, the **same file** `src/adapter-v20.js` is
locked to.

| Measured (M2 Pro, two worms moving, shooting and respawning) | Result |
| --- | --- |
| One core | 460k ticks/s = **7,700× real time** (1.3M–2M on an empty map) |
| Six processes | 340k–370k each, **about 2.2M ticks/s ≈ 36,000×** |
| Determinism | the same seed twice → identical positions and health after 20,000 ticks, to six decimals |

At a frameskip of four that is **550,000 agent steps a second**. Ten million
steps is a few minutes as far as the environment is concerned.

### Input is one bitmask and two messages

> **Corrected 2026-09-20.** This document first recorded bit 256 as "change
> weapon". That was wrong. In `worm.tw()` bit 256 calls `worm.nv()`, which
> carves away the two points 2 and 4 pixels in front of the worm — it is
> **dig**. Changing weapon is a message, not a bit.
> `test/env-engine.test.js` now watches the dirt disappear and the slot change,
> separately.

Write `worm.Wa` and call `world.update()`; that is the whole of it.

`1 left | 2 right | 4 aim up | 8 aim down | 16 fire | 32 jump | 64/128 rope length | 256 dig`

**Jump (32) and dig (256) fire on the press, not while held.** The engine
remembers whether the key was down last tick and does nothing until it has seen
it released. A policy that holds either bit down jumps exactly once.

Two things are not bits, and a real room sends them separately too.

- **Rope** — throw `worm.kx(world)` / release `worm.Pw()`. One network command
  (`Cb.hm`).
- **Weapon change** — `worm.oq(worm.Ka + offset)`. It is a **relative** move, and
  the network command (`Ab.offset`) is relative too. On spawn, `oq(a.Bf)`
  restores the slot the player last had.

A live room does exactly the same thing: `room.ea` copies each player's `Lb`
onto `worm.Wa` and calls `world.update()`. **A policy trained on these maps one
to one onto real key input**, which is why the action in `src/env/actions.js` is
kept as three pieces, `{ keys, rope, weapon }`. Folding them into one bitmask
would only mean splitting them again on the way into a live game.

### The engine generates maps, reproducibly from a seed

The generator a room uses when its host asks for a random map is right there:
`level.Jp(rng, settings, 504)` gives a 504×350 "Random Dirt". **Training needs
no `.lev` file at all.**

One catch. The Perlin permutation that shapes the terrain is shuffled with
`Math.random()`, so a seed alone does not reproduce a map. `randomLevel(seed)`
in `src/env/engine.js` lends the generator a seeded `Math.random` for the call
and gives the real one straight back, so the same seed gives the same map down
to the byte.

And **the `.lev` files in the kit make digging meaningless**: Simple2 has zero
pixels of dirt (all rock or air) and Arena_1 only 1,457 of 176,400. A generated
level is 99,223 — over half. If the point is to learn the Liero where you dig
through the ground, generated levels are the right ones. Generating one costs
about 7 ms, which is a noticeable share of a short episode, so make a pool of
them up front and cycle it (`--level-pool N`).

### What a trained model costs to run

Batch of one, one torch thread — the same conditions as a live game:

| Model | Size | One decision | At 60 Hz |
| --- | --- | --- | --- |
| 0.63M parameters | 2.4 MB | **0.075 ms** | 0.45% of one core |
| 2.45M | 9.4 MB | 0.145 ms | 0.87% |
| 9.66M | 36.9 MB | 1.07 ms | 6.4% |

**Inference will never need a GPU.** A paid model call in the earlier repository
took 3.5–4.4 s per decision, so this is roughly fifty thousand times faster.

### What training costs (forward + backward + Adam, batch 1024)

| | 1 CPU thread | 10 CPU threads | MPS |
| --- | --- | --- | --- |
| Terrain conv policy | 11,532/s | 5,984/s | **108,731/s** |
| Vector observation only | **696,713/s** | 478,621/s | 1,209,578/s |

- **A GPU is optional.** On a vector observation, one CPU thread is faster than
  the environment can feed it.
- For a convolution, MPS is ten times quicker. Even so, CPU alone is fifteen
  minutes for ten million steps.
- **Giving torch more cores makes it slower** — ten threads are half the speed
  of one. Pin torch to one or two and leave the other six to eight to the
  environment workers, which are the CPU-bound half.

## 2. Loading the engine — repository code and the kit

`src/env/engine.js` reads the bundle itself and evaluates it with
`vm.runInThisContext`. **It does not depend on the kit's `probe.mjs` or its
patched `engine.js`.** All it needs are the four downloaded assets, exactly the
ones `fetch-assets.sh` fetches.

- Before running anything it compares the SHA-256 of `game-v20.orig.js` against
  `CLIENT_SHA256` in `src/adapter-v20.js` and refuses if they differ. The
  physics only transfer while it is the **same file** the dashboard reads.
- The patch happens in memory only: the trailing `u.js()` (the UI boot) is
  replaced by one line that hands the closure's classes to `globalThis`. No file
  is written, so no black box is left behind.
- The classes come out named (`World`=`Pa`, `Worm`=`V`, `Level`=`Z`, `Rng`=`eb`,
  `Zip`=`ib`, `Mod`=`U`, `Sprites`=`Jc`, `Wasm`=`A`, `Reader`=`H`).
- One engine per process. The bundle installs itself on the globals, so opening
  a second directory would quietly give you the first one's classes;
  `loadEngine()` memoises for that reason.

### The kit (`artifacts/headless-sim/`, gitignored)

Kept for benchmarks and poking around.

```sh
cd artifacts/headless-sim
node patch-engine.mjs  # game-v20.orig.js -> engine.js (skip if the assets are there)
node boot.mjs          # boot the engine, make a world, two worms
node fight.mjs         # throughput
node determinism.mjs   # the same seed twice
node obs-demo.mjs      # both observation designs printed from a real map
./fetch-assets.sh      # download the assets again
```

The torch benchmarks want a venv (`artifacts/` is gitignored, so it can live
inside):

```sh
cd artifacts && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy
.venv/bin/python headless-sim/infer-torch.py
.venv/bin/python headless-sim/train-cost.py
```

Worth knowing:

- The assets come from the **versioned path**: `res.dat`,
  `vendor/wasm-flate.wasm`, `vendor/json5.min.js` and `game-min.js` under
  `https://www.webliero.com/v/20/`. The site root (`/res.dat`) is a 404. json5
  declares `var JSON5`, so append `globalThis.JSON5=JSON5` when evaluating it.
- Class map: `Pa`=World (zero-argument constructor, `update()`, `Of(other)` deep
  copy, `reset(seed)`), `V`=Worm, `Z`=Level (`read(name, arrayBuffer)` for a
  176,400-byte .lev), `U.dj`=the mod json5 parser, `ib.read`=zip,
  `A.Pc(url)`=the wasm loader, `Jc.read`=sprites.
- Settings are `U.dj(mods/liero133/mod.json5)` plus sprites, then `.normalize()`,
  assigned to `world.s`. liero133 is stock Liero 1.33 with 40 weapons.
- Spawning: `world.ox(color, ownerId, [five weapon ids])`.
- **Respawning is not `worm.nx()` alone.** A dead worm is dropped from
  `world.za` by the tick that kills it and `nx()` does not put it back. It takes
  `u=true` → `nx(world, loadout)` → `za.push(worm)` for that worm to be
  simulated again. `respawnWorm()` in `src/env/engine.js` is that.
- Room settings are world fields: `qd`=bonusDrops (default 2),
  `Pe`=bonusSpawnFrequency (1800 ticks), `mf`=weaponChangeDelay,
  `Te`=damageMultiplier, `le`=loadingTimes (0.4). Training defaults to
  `bonusDrops: 0`, because otherwise where a medkit fell decides the match.
- `world.reset(seed)` empties worms, projectiles and the tick counter but leaves
  the level alone. Undoing the digging takes `world.level.Of(pristine)` as well.
- `world.Ga` (events and sound) and `world.$p` (death) may stay null; every call
  site null-checks them.

## 3. The observation, as built

Mixed, as recommended: state in a vector, terrain and projectiles through a
small convolution. Both live in `src/env/observation.js`, and
`observe(view, into, kinds)` builds only the ones asked for.

### The vector (`observationSpec()` sizes it to the player count)

| Block | Count | Contents |
| --- | --- | --- |
| `rays` | 16 | clockwise from due right, distance to the first solid pixel / 120 |
| `health` `velocity` `aim` `facing` | 6 | health, px/tick, aim as cos and sin, which way it faces |
| `contacts` `stepping` | 5 | the engine's own probe counts on four sides, and its automatic step-up |
| `walkLeft` `walkRight` | 8 | one-hot clear/step/dirt/rock |
| `weapons` | 15 | five slots × (ammo left, ready to fire, selected) |
| `rope` | 5 | out, attached, where it is, how long |
| `foes` | 9 per foe | each: alive, where, how far, which way, health, speed |
| `projectiles` | 12 | the three nearest shots: where and where to |

**The size follows the player count**: 67 solo, 76 for a duel, 85 for three, 103
for five. Foes fill the slots nearest first and the rest are zeros. Set
`observationFoes` higher than the count and **the same policy plays a duel and a
five-way** — the unused slots are simply zero.

Showing only one opponent in a three-way is not enough. The worm shooting at you
and the worm you are shooting at are usually not the same one, and a policy that
can see only one of them cannot make that choice at all.

Aim goes in as cos and sin rather than an angle, so the wrap at π is not a
cliff. Distances are divided by 300 px and clipped, with the direction kept
separately as a unit vector so that clipping never hides which way something is.
**Own projectiles are included** — a worm's own explosions are close to half the
damage it takes, so there is nothing to gain by hiding them.

### The terrain patch — 1,024 bytes stored, 4 × 32 × 32 into the convolution

*Superseded: six channels of 121 × 213, the worm's whole view. Worms show up in
the picture as well as in the vector. See `PATCH` in `src/env/observation.js`.*

Rock / dirt / free space / projectile. One cell stands for 2×2 real pixels and
answers for **the hardest thing in them**, so a wall one pixel thick cannot fall
between two samples and read as open air. Off the level reads as rock, which is
how the boundary behaves. The first three channels are exclusive, so exactly one
of them is 1 per cell — which is what keeps "blocked" and "not measured" apart.

**It is stored and sent as one byte per cell**: the low two bits are the
terrain, the third bit a shot. Expanding to four one-hot planes is free on the
GPU, so it happens there. Sending one-hot floats would be four times the bytes
for the same information, and a training run would push tens of gigabytes
through a pipe for nothing.

### What each costs (measured)

| | One observation | Environment | Training end to end |
| --- | --- | --- | --- |
| Vector only | 0.003 ms | 76k steps/s | **11,200 steps/s** |
| Vector + patch | 0.017 ms | 37k steps/s | **6,570 steps/s** |

**The patch costs about ten times the vector.** Mixing them is the default —
that is the design — and `--no-patch` is there for a quick look. Ten million
steps is 25 minutes mixed, 15 minutes on the vector alone.

### What keeps the live-game vector identical

The observation encoders read neither the engine nor the adapter. Both are
turned into **one view** in `src/env/view.js` first, and only that is read.

- `viewFromWorld(world, self, foes)` — from a headless world
- `viewFromSnapshot(state, terrain)` — from the live game's `/state` and `/map`

The field names are the adapter's own. `test/env-observation.test.js` drives the
entire encoder **through the live path** (fixture → `snapshotV20` → view →
vector), so if what the adapter reports ever stops lining up with what the
encoder wants, a test fails rather than a first real match going quietly wrong.

The contact counts and walk probes are also inlined in the adapter, because that
function is stringified into the page where module scope does not exist. The two
copies are deliberate, and a test runs both against the same fixture.

## 4. The structure

- **Environment**: `WormEnv` in `src/env/env.js`. Built — section 4-1. The
  player count is one setting, and 1, 2, 3, 5 and 8 all run on the same code.
- **Reward**: built (`src/env/reward.js`). Section 4-0.
- **Training**: PyTorch PPO with self-play. Built — section 4-3. (A league of
  past snapshots is not.)
- **Bootstrapping**: the earlier repository has recorded human play
  (`/Users/dongho/projects/my-first-ai-wormy/artifacts/`, rope work included).
  Imitating it first would skip past skills like the rope that random
  exploration does not find on its own.
- **Layout**: six to eight CPU cores for the environment, one or two torch
  threads plus MPS for the learner.
- **Into a live game**: the physics are the same code, so there is no gap there.
  What is left is network delay and key sampling, so training mixes in an input
  latency of 0–3 ticks — `inputLatencyTicks: [0, 3]` already does it. Deployment
  could be (a) a Python policy over a socket, (b) ONNX inside the Node driver,
  or (c) the weights in the page itself. All three are possible.

### 4-0. The reward — who hit whom comes first

With three or more worms, **"the other one lost health" is not a signal.** Two
of them can be fighting while the third watches and the number is the same. So
the engine is asked directly.

- Every hit goes through `worm.ud(world, amount, attackerId, weaponId)`.
  `src/env/engine.js` wraps that one function and records **the amount actually
  applied** (the health either side of the call) along with the attacker. The
  original runs untouched; the wrapper only reads.
- Kills need no wrapper. `world.$p` is a hook the engine already calls.
- Own explosions and falls arrive with the worm's own id — counted as damage
  taken and not as damage dealt, which is exactly the accounting wanted.

A test holds the conservation law: **everyone's damage taken minus everyone's
self-damage equals everyone's damage dealt.** (It failed the first time, because
a shotgun's spread also caught the third worm that was only watching. Without
attribution that damage would have been credited to the wrong worm.)

The terms and their default weights:

| Term | Weight | Meaning |
| --- | --- | --- |
| `damageDealt` / `damageTaken` | ±1/100 | 100 health is one point |
| `kill` / `death` | ±3 | finishing beats wounding, and dying is expensive |
| `explore` | +0.02 | a 16 px cell entered for the first time |
| `revisit` | −0.02 | back to a cell it was recently in: going in circles |
| `stuck` | −0.01/step | under 14 px of movement across 60 steps (four seconds) |
| `goalProgress` / `reachedGoal` | +0.02/px, +2 | only when a goal is set |

*Superseded.* Dealing damage now pays twice what taking it costs and a kill
twice what a death costs, because at equal weights combat sums to exactly zero
between worms sharing one policy and the run duly learned to stop firing.
Covering ground pays as a share of the map rather than a flat amount per cell,
and pointing at somebody you could hit pays a little. `DEFAULT_WEIGHTS` in
`src/env/reward.js` is the list, with the measurement behind each one.

**Being stuck and going in circles are different failures and need different
measurements.** Standing still is a displacement across a window of time. Going
in circles is covering ground and arriving nowhere, which only a memory of where
it has already been can see. `src/env/progress.js` measures both.

The stuck penalty has a trap in it. **Make it large and the optimal policy is
suicide** — dying and respawning becomes cheaper than digging out of a hole.
Hence 0.01 per step: a worm stuck for the whole of a 900-step episode loses 9,
which is three deaths, and suicide only pays after more than about 300 steps
stuck, by which point it really has dropped out of the game. A test pins that
inequality.

### 4-1. Using the environment

```js
import { loadEngine } from "./src/env/engine.js";
import { WormEnv } from "./src/env/env.js";
import { KEYS, ROPE } from "./src/env/actions.js";

const engine = await loadEngine();              // once per process
const env = new WormEnv(engine, {
  agents: 3,                                    // 1 is solo, 5 a brawl
  observationFoes: 4,                           // pin the vector to four foes (optional)
  frameskip: 4,                                 // deciding at 15 Hz
  episodeTicks: 3600,                           // one minute of game time
  inputLatencyTicks: [0, 3],                    // for the move to a live game
  loadout: "random",                            // five of the mod's forty, per episode
  observations: ["vector", "patchBytes"],
  rules: { bonusDrops: 0 },
  level: (engine, seed) => pool[seed % pool.length],   // omit for a new map each episode
});

const { observations } = env.reset({ seed: 11 });
const { rewards, done, info } = env.step([
  KEYS.right | KEYS.fire,                       // a bitmask is enough
  { keys: KEYS.left, rope: ROPE.throw, weapon: 1 },   // or the whole action
  actionFromHeads(heads, at),                   // or a policy's seven heads
]);
```

- `observations[i]` is `{ vector, patch }` and **the buffers are reused**. Copy
  anything that has to survive the next step.
- `info.events[i]` is that step's `{ damageDealt, damageTaken, killed, died }`;
  `info.totals[i]` is the episode so far.
- The reward is read **before anyone respawns**, or a death looks like a worm
  that healed back to full.
- One seed fixes the map, the world RNG, the weapons and the input latency. The
  same seed and the same actions give the same episode, and
  `test/env-engine.test.js` compares to six decimals.

### 4-2. The action is seven heads

A policy emits seven small choices, not one of 648.

`move(3) · aim(3) · fire(2) · jump(2) · dig(2) · rope(3) · weapon(3)` → **18 logits**

Left and right together is the engine doing nothing, so they are one three-way
choice rather than two bits that can contradict each other; the same for aiming.
A policy that has learned to walk right keeps that when it learns to fire.

### 4-3. The trainer

```sh
npm run train -- --agents 3 --total-steps 8000000     # these are already the defaults
npm run train -- --help
```

- **N Node workers and one Python process.** The engine is JavaScript and the
  fastest thing here for a convolution is PyTorch on MPS, so the two meet on a
  worker's own stdin and stdout: a four-byte length and then binary, with the
  first frame describing the layout of all the rest as JSON. No port to pick, no
  socket to clean up, and the workers die with the parent.
- **Every worm shares one policy.** In a free-for-all that is self-play by
  construction: whatever one of them learns comes straight back at it.
- 0.52M parameters. The vector goes through a dense layer, the patch through two
  convolutions, then a 256×2 trunk, seven heads and a value.
- The observation normaliser (running mean and variance) is saved with the
  checkpoint. Raw observations mix ratios, pixel counts and velocities, and
  without it most of early training goes into discovering each one's scale.
- The best policy is kept separately from the latest one, on a smoothed episode
  reward, because self-play wanders and the last checkpoint is not the best one.

Measured (M2 Pro, 4 workers × 8 worlds × 3 worms = 96 worms at once):

| | Training throughput | Ten million steps |
| --- | --- | --- |
| Vector + patch | 6,570 steps/s | about 25 minutes |
| Vector only (`--no-patch`) | 11,200 steps/s | about 15 minutes |

An update spends roughly 12% waiting on the workers, 60% collecting the rollout
and 40% on the gradient steps. Collection is large because it runs 128 forward
passes on a batch of only 96; raising `--envs` amortises that.

**Worth knowing:** episodes end on their tick budget, which is strictly
truncation, but it is treated as done without bootstrapping. With 128-step
rollouts against 450-step episodes that is a couple of samples per update.

### 4-4. Watching it play

Numbers say whether a policy is improving. They do not say whether it looks like
someone playing, and the only way to know that is to watch.

```sh
npm run watch                        # the newest run's best policy
npm run watch -- --agents 5 --speed 2
```

There is a **Watch** button on the training page that does the same for the
selected run and opens it. One match is played at the speed the game actually
runs at and drawn on port 8769.

It is **the headless engine rendered, not a room on webliero.com.** The physics
are identical — the same bundle, the same checksum — but putting a policy into a
live online room still needs the key-input path from section 0.

### 4-5. Watching the run itself

Training runs for hours, and there has to be somewhere to see whether it is
going well.

```sh
npm run train -- --agents 3 --total-steps 8000000   # a line per update
npm run monitor                                     # http://127.0.0.1:8768
```

- Runs are written to `artifacts/runs/<id>/` as `run.json` and `metrics.jsonl`.
  Append-only text, so a run that is killed halfway is still readable, and the
  monitor reads **only the bytes that have appeared** since it last looked. The
  trainer does not need to know anyone is watching. JS
  (`src/train/recorder.js`) and Python (`train/run.py`) write the same format;
  the checkpoints sit in the same folder.
- **Every numeric field becomes a chart.** When the trainer started logging
  `policyLoss`, it appeared without the page being touched. Known names get a
  label and a direction, so rises and falls read as green or red.
- It is a **separate page on a separate port** from the game dashboard (8766).
  Training needs no browser and the dashboard needs no training. The monitor
  reads runs; the one thing it can write is starting a viewer.

## 5. Stages

1. ~~environment wrapper, observation and reward, determinism tests~~
   **done 2026-09-20.**
2. **Walking to a point** — the environment is ready. The `goals` option gives
   each worm a target and turns on the `goalProgress` / `reachedGoal` reward. It
   has not been run as a task yet.
3. ~~combat self-play~~ **done 2026-09-20 — as a three-way free-for-all.**
   `npm run train`. Why N-way rather than 1v1 is in section 4-0, attribution.
4. Into a live game — the key-input path has to exist first (section 0).

### The throughput that actually came out (M2 Pro)

Environment alone, one core, two agents:

| | Agent steps/s | Speed-up |
| --- | --- | --- |
| Vector only | 76,485 | 2,549× |
| Vector + terrain patch | 36,509 | 1,217× |

With training, 4 workers × 8 worlds × 3 worms (96 worms at once):

| | Training steps/s | Ten million steps |
| --- | --- | --- |
| Vector + patch | 6,570 | about 25 minutes |
| Vector only | 11,200 | about 15 minutes |

## 6. Decided, and still open

**Decided (2026-09-20):**

- **The trainer is Python and PyTorch on MPS; the environment is Node workers.**
  Training in JavaScript does not work for a convolution — an MLP forward pass
  alone is 0.138 ms, and a convolution with a backward pass is hundreds of times
  that. The two are joined by a binary pipe, and it costs 12% of an update.
- **N-way free-for-all rather than 1v1.** The count is a setting and 1, 2, 3, 5
  and 8 run on the same code.
- **No hand-written opponent.** Everyone is the same policy, so self-play is
  free.

**Still open:**

- A league of past snapshots — right now it only ever faces its own latest self.
- Whether to actually run the walking task (stage 2) or stay on combat.
- Whether to bootstrap from recorded human play, for skills like the rope that
  random exploration does not find.
- Whether to bring the earlier repository's reflex and Jev players in as
  opponents and baselines, or ignore them.

## 7. House rules for this repository

- **English only** — the dashboard, the code, the comments and the documents.
- **No push, no PR, no CI** unless the message asks for it explicitly. This was
  the user's clear policy in the earlier repository and it is the same project.
  Verify with `npm test` locally and keep commits local.
- Behaviour changes are **measured here**, not handed to the user to try.
- Ports: dashboard **8766**, fixture preview **8767**, training monitor
  **8768**, match viewer **8769**, Chromium CDP **9334** — chosen not to collide
  with the earlier repository's 8765 / 9333. The Chromium profile is shared at
  `~/.cache/wormy-ii/chrome-profile`.
- **Never call `browser.close()`** on a browser attached over CDP: the room dies
  and the next one comes with a CAPTCHA. Closing the last page does the same.
- Read `.ai-memory` before starting. The findings are under
  `game.headless_engine` and `learning.model_cost`; the environment, trainer and
  monitor under `env.wrapper`, `training.ppo` and `training.monitor`.
- torch lives in `artifacts/.venv` (gitignored). If it is missing,
  `npm run train` says how to make it:
  `cd artifacts && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy`
- Node 26 (`.nvmrc`).
