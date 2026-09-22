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

## What follows from this

1. Goals at a distance the policy can reach by walking first, then farther,
   as the earlier note planned. The environment needs a goal maker that takes
   a radius; the observation and the network need nothing.
2. Either drop the five rope-hostile maps from the movement stage, or give the
   patch a channel for "the rope holds here" — `flags & 7` — so the policy can
   tell.
3. Log the entropy per head. With eight heads summed into one number, a rope
   head that has gone deterministic and a fire head kept uniform by the bonus
   read the same.
4. Fix the release press in `keys.js` before the next live match.
