# What the movement runs learned, and what stops them

Two runs on the `apac-jp-stage` cluster, started 2026-09-22 06:48 UTC on
`sha-b19c846`, `--task movement --agents 6 --patch-scale 4`, no opponents,
entropy coefficient fixed at 3e-4. The controlled variable was
`--rope-cooldown`: `a` hears every rope message (0), `b` one every ten
decisions.

Everything below was measured on 2026-09-22 between 13:00 and 14:00 UTC, from
the pods' logs (the trainer's per-update line) and from the engine on a laptop.

## The curve

`goals` is destinations reached per worm per one-minute match, averaged over
the six worms and over the updates in each two-million-step bucket. `rope` is
throws asked for / decisions spent attached, out of 900 decisions.

| steps (M) | a goals | a rope    | a entropy | b goals | b rope   |
| --------: | ------: | :-------- | --------: | ------: | :------- |
|       0-2 |    0.47 | 265 / 157 |      7.46 |    ~1.0 | 56 / 398 |
|       4-6 |    3.40 | 229 / 419 |      6.54 |    ~2.4 | 64 / 443 |
|      8-10 |    4.56 | 297 / 383 |      5.79 |    ~3.2 | 65 / 391 |
|     12-14 |    5.40 | 333 / 343 |      4.54 |    ~4.3 | 50 / 390 |
|     18-20 |    5.88 | 304 / 374 |      3.79 |         |          |
|     26-28 |    5.98 | 303 / 379 |      3.02 |         |          |
|     34-36 |    6.35 | 285 / 376 |      2.68 | 5.5-6.0 | 60 / 470 |

The maximum entropy of the eight heads is 7.57. Both runs start at it — a
uniform policy throws on a third of its decisions, which is the 300 throws a
match `a` opens with — and both are still slowly sharpening at 36M steps, with
the KL held at 0.016 and the rate at 2e-4.

So the run learned fast to about 10M steps and has gained roughly one goal a
match over the 25M since. `b` is a little behind `a` the whole way: hearing
fewer rope messages did not help.

## What the numbers mean: baselines in the same world

Measured with `train/workers.py` driving the exact cluster configuration (six
worms, the 18 community maps, input delay 6-21 ticks, the movement weights,
fire locked), 48 matches each, no network:

| policy                                   | goals / match | stuck steps | rope        |
| ---------------------------------------- | ------------: | ----------: | :---------- |
| still                                    |          0.01 |         809 | 0 / 0       |
| uniform random over the eight heads      |          0.15 |          78 | 299 / 174   |
| walk toward the goal, jump when blocked  |          0.36 |         498 | 0 / 0       |
| the same, plus a hand-written rope rule  |          0.41 |         453 | 38 / 271    |
| **run `a` at 36M steps**                 |      **6.35** |          26 | 285 / 376   |

The policy is forty times the random policy and fifteen times a scripted
walker. It is not failing to learn; it learned one way of getting about and
has been polishing it. What that way is can be read off the engine.

## What the rope actually is in this mod

`settings.pb` of csliero rewormed v0.37, and `zc.update` in the bundle:

- A throw leaves at **8 px a tick** along the aim, so it crosses 100 px in
  13 ticks — within one decision at frameskip 4 — and flies up to 250 px
  (`Gj`) before being pulled back.
- On touching terrain the length is set to **28 px** (`th`), whatever the
  distance. The worm is then farther than the length, so every tick it is
  accelerated toward the anchor by **0.083 px/tick²** (`Fj`): after a second
  of hanging it is moving at 5 px a tick. The rope is a grappling hook that
  reels the worm in on its own. Nothing has to be shortened.
- Lengthening pays out 10.6 px a tick up to 250 (`wh`, `vh`); shortening
  takes it back. Releasing keeps the worm's velocity. Throwing again while
  attached drops the anchor and starts a new throw from where the worm is.
- For scale: walking is **0.565 px a tick** (34 px a second), a jump rises
  22 px, and holding an aim key turns the aim 9° in ten ticks and 90° in
  forty.

Goals are drawn uniformly over the map. The mean distance between one goal and
the next is 450-550 px on these maps (693 on cs_4004). Walking that in a
straight line takes 14 seconds, so walking alone tops out near four goals a
minute even on open ground — and the scripted walker shows the ground is not
open: it spends half its match stuck against a wall. Six goals a minute is
only possible on the rope.

With `--rope-cooldown 0` a throw is followed by another one 1.5 decisions later
on average (measured in `env.js`), so a rope is never held long enough to
swing; but a stream of re-throws along the aim is itself a way to fly — every
throw re-anchors ahead and the pull follows. That is what the viewer shows as
spamming. It works: it is the six goals. What it cannot do is anything that
needs the rope held, aimed and let go at a moment — which is the part a person
would call using the rope.

## What the policy cannot see

The mod's material table has 256 entries: 175 with no flags at all, 9 with
only bit 5, 23 rock (bit 2), 36 dirt (bits 0-1), 14 background (bit 3). The
worm treats anything without bit 3 as solid — it stands on it and walks into
it. The rope (`zc.update`) attaches only where bits 0-2 are set. **A wall
drawn in a no-flag colour stops the worm and lets the rope through** — it
flies on to whatever is behind, or to the map edge. Verified in the engine
with a flag-0 ceiling: the rope passed through it and attached at y = -1.

Share of solid pixels the rope can attach to, per community map:

| map                   | attachable |
| --------------------- | ---------: |
| cs_ash3_remix         |      37.7% |
| cs_ztn3tourney1c      |      47.8% |
| cs_cat                |      51.2% |
| cs_we_ll_see          |      69.5% |
| cs_replaced           |      81.1% |
| the other twelve      |     96-100% |

The observation encodes each pixel as rock, dirt or free by bits 3 and 0-1,
so those walls read as rock in the patch, in the map and in the rays. On five
of the seventeen maps the policy is shown a wall it can hook and cannot.

## The curriculum that was planned and the one that ran

`runs-2026-09-22.md` planned near goals at the same height first, then
diagonals, then far goals. `env.js` has one goal maker, `groundedGoal`, which
is uniform over the whole map, and `--task movement` uses it. The runs started
at stage three. Nothing in the observation says how far a goal is beyond one
saturating distance, and nothing in the reward pays for holding a rope; the
only route from six goals to twelve is discovering aimed, held, released throws
by chance under a policy whose entropy is a third of maximum.

## The physics are the same; the keys are not

The bundle is the one the room runs, so the world is the same. Two things
about input differ, read off `Kc.qo` and `Cb.apply` in the bundle:

- The client's `NinjaRope` key always sends **throw** (`hm = true`). Release
  is the **Jump** key while ChangeWeap is not held (`hm = false`). Jump also
  releases the rope in the real game, and the environment's jump head does
  not.
- `src/live/keys.js` `pressesForAction` presses `NinjaRope` for a release as
  well as for a throw, so in a live room a policy's release is a re-throw.

Neither touches training; both matter the first time a policy plays a room.

## What was changed on 2026-09-23, before the next runs

1. **Goals within reach first.** `--goal-radius 96-1600` draws a destination
   within 96 px of the worm and moves the limit out to 1,600 px over the first
   `--goal-grow` (0.6) of the run; a goal not reached in `--goal-patience`
   (450) decisions is given up on and counted as `goalsMissed`. The radius the
   worlds are drawing at is logged as `goalRadiusPx`.
2. **A price per throw.** `--rope-throw-cost` charges per rope throw the engine
   hears (`fromRopeThrow`). Holding on is free. The a/b of the next runs is
   this knob: 0.01 against 0.
3. **The rope-transparent ground is its own kind.** `terrain.js` classifies
   ground by the flags the engine tests — background, dirt, rock, and the
   flagless `ghost` that stops a worm and lets a rope through — and the patch
   has eight planes (rock, dirt, free, ghost, shot, foe, self, goal) and the
   map five channels (free, dirt, rock, ghost, occupants). The goal is drawn on
   both when it is in view. Checkpoints and worker layouts carry the channel
   counts, so a policy from another picture is refused rather than misread.
4. **One entropy per head** in the metrics (`entropyMove` … `entropyWeapon`)
   and on the log line.
5. **`best.pt` of a movement run is picked on `goalsReached`** (`bestGoals`),
   not on the combat score, which is zero throughout — the first checkpoint
   ever written used to stay "best" for the whole run.
6. **The normaliser is updated after the gradient step**, so an update scores
   its rollout under the statistics the rollout was collected with.
7. **Letting go of the rope is a jump press**, in the environment and in the
   live driver, which is what the client does. A jump pressed while the rope
   is out lets go of it.
8. The probe against still worms is off in the movement task; there is nothing
   to kill.

Baselines under the new world are the same as above to within noise: the
pictures changed, the rules a scripted walker follows did not.

## The next runs, 100 minutes in

Started 15:18 UTC on `sha-d325732`, a with `--rope-throw-cost 0.01`, b with 0,
both on the clock schedule. Per half-million steps:

| steps (M) | radius | a goals / missed | a rope    | b goals / missed | b rope    | b jump entropy |
| --------: | -----: | :--------------- | :-------- | :--------------- | :-------- | -------------: |
|   0.5-1.0 |    115 | 4.7 / 1.1        | 158 / 146 | 5.3 / 1.0        | 280 / 155 |           0.69 |
|   1.0-1.5 |    127 | 5.5 / 1.0        |  94 / 89  | 6.1 / 1.0        | 284 / 118 |           0.69 |
|   3.0-3.5 |    177 | 3.4 / 1.1        |  40 / 17  | 3.6 / 1.1        | 279 / 103 |           0.65 |
|   5.0-5.5 |    228 | 2.5 / 1.2        |  12 / 5   | 2.9 / 1.1        | 240 / 143 |           0.63 |
|   7.0-7.5 |    278 | 2.0 / 1.2        |  22 / 14  | 4.3 / 0.7        | 191 / 312 |           0.29 |

Two things, both measured:

- **The throw cost killed the rope.** At 0.01 a throw costs what a third of a
  near destination pays, before the rope has earned anything; the gradient's
  answer was to stop throwing, and by 4M steps `a` threw eleven times a match
  and walked. Charging for a thing before it is useful removes the chance of
  it becoming useful. `a` is restarted without the cost.
- **The clock outran the policy.** Both runs reached five or six destinations
  a match at 115-130 px, and three by the time the radius had moved out to
  250, while nothing in a schedule that grows with the steps could notice. `b`
  recovered on its own once its rope head started holding (312 decisions a
  match on the rope at 7.3M, 1.6 per throw against 0.3 at first), but a
  curriculum should not depend on that. `--goal-curriculum success` moves the
  radius on the share of destinations reached instead; `a` runs it, `b` stays
  on the clock as the control.

## Why no rope was ever held, traced throw by throw

The local run at a fixed 120 px radius (1M steps, nine destinations a match)
was played back with every rope throw followed, the input delay accounted for.
Per worm and match:

| | |
| --- | ---: |
| throw choices | 95 |
| release choices | 275, of which with no rope out | 214 |
| jump presses | 221 |
| decisions with a rope out | 58 of 900, being pulled up on 11 |
| median time a rope was out | 2 decisions (0.13 s), 8 or more on 2.4% |
| throws aimed above 60° | 87% |

Ropes were let go by the policy's own release 93% of the time, by a jump 7%.
The release choice had to press Jump to match the client, so it became a
second jump key: 78% of its presses were with no rope out. And even without
that, a choice made afresh fifteen times a second keeps a rope for twelve
decisions with probability near zero, so the pull — which needs about that
long to move the worm a hundred pixels — was never experienced and never
learned. The aim was not the problem: seven throws in eight went up.

Changed on 2026-09-23, both runs restarted:

- **No release choice.** The rope head is throw or nothing; letting go is the
  jump head, as it is a person's Jump key.
- **A throw is a commitment** (`--rope-hold`, `RopeHold` in actions.js): for
  that many decisions another throw is ignored and the jump key dropped, in
  the environment and in the live driver alike. `a` holds 12, `b` 24.
- **The worm sees what a throw would hook** (`ropeReach` in the vector): along
  the aim, the distance to the first ground that holds a rope and how far up
  it is. A mechanism it can look at rather than one to find blind.
- **Half the destinations are above the worm** (`--goal-above 0.5`, at least
  48 px up), where a jump does not reach, so the success curriculum cannot
  move on until the rope is used.

## The same trace under a twelve-decision hold

A local run on the changed interface (`--rope-hold 12`, the success curriculum
from 96 px, half the destinations above) was played back the same way at 0.6M
steps, one hour in. Per worm and match:

| | |
| --- | ---: |
| throw choices | 542 (the head is still near uniform) |
| throws that went through | 66 — one every 13-14 decisions, the hold plus a coin flip |
| jump presses | 225 |
| decisions attached | 695 of 900, being pulled up on 147 |
| median time a rope was out | 39 decisions (2.6 s), 31 or more on 54% |
| throws aimed above 60° | 70%; above 30°, 76% |
| height gained while out | +19 px on a throw above 60°, −46 px on one below −30° |

Ropes are let go by a jump 89% of the time; the rest lose their anchor (the
dig key is random too) or end with the match. So the hold does what it was
for: a rope is kept for 39 decisions in the median instead of 2, and the pull
is felt on 147 decisions a match instead of 11. What it has not done yet is
turn into destinations — 2-3 reached a match at 96-101 px, against 4.7-5.3 at
115 px for the earlier runs at the same age, whose destinations were all on
the ground. A worm hanging under a ceiling for three seconds at a time is
slower to the ones on the floor; whether it learns to hang only for the ones
above is what the two cluster runs are for.

Both restarted at 04:30 KST on 2026-09-23 on `sha-3066320`, `a` holding 12
and `b` 24, and their first matches read the same way: 0.8 and 0.7 reached at
96 px, 65 and 35 throws, 693 and 789 decisions attached.
