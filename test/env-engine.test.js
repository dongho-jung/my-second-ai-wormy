// The half of the environment that needs the real game engine.
//
// The bundle is not in the repository — `artifacts/` is gitignored, because the
// upstream's file is theirs to ship — so these skip with a reason rather than
// fail when the kit is not there. Everything that can be checked without it is
// in env-observation.test.js and env-reward.test.js, which always run.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { CLIENT_SHA256 } from "../src/adapter-v20.js";
import {
  ASSETS,
  DEFAULT_ENGINE_DIR,
  loadEngine,
  respawnWorm,
} from "../src/env/engine.js";
import {
  ALL_KEYS,
  EDGE_TRIGGERED,
  KEYS,
  ROPE,
  applyAction,
  describeKeys,
  packKeys,
  unpackKeys,
} from "../src/env/actions.js";
import { WormEnv } from "../src/env/env.js";
import { PATCH, VECTOR_OFFSETS, encodePatch, encodeVector } from "../src/env/observation.js";
import { viewFromWorld } from "../src/env/view.js";
import { diggableAt } from "../src/env/terrain.js";

const absent = Object.values(ASSETS).filter(
  (name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)),
);
const skip = absent.length
  ? `no engine kit in ${DEFAULT_ENGINE_DIR.pathname}: missing ${absent.join(", ")} — run artifacts/headless-sim/fetch-assets.sh`
  : false;

const LOADOUT = [0, 2, 3, 5, 10];

/** A world with two worms placed by hand, for testing one thing at a time. */
async function arena({ seed = 4242, level } = {}) {
  const engine = await loadEngine();
  const world = engine.createWorld({
    level: level ?? engine.randomLevel(seed),
    seed,
    rules: { bonusDrops: 0 },
  });
  const worms = [0, 1].map((playerId) =>
    engine.spawnWorm(world, { color: playerId, playerId, loadout: LOADOUT }),
  );
  return { engine, world, worms };
}

/** Settle a worm on the ground where it stands, with no keys held. */
function settle(world, worms, ticks = 40) {
  for (let tick = 0; tick < ticks; tick++) {
    for (const worm of worms) worm.Wa = 0;
    world.update();
  }
}

/** Somewhere every pixel within `radius` is dirt a worm can dig. */
function findDirt(world, materialFlags, radius = 8) {
  const level = world.level;
  const terrain = {
    data: level.data,
    width: level.width,
    height: level.height,
    materialFlags,
  };
  for (let y = radius; y < level.height - radius; y += 2) {
    for (let x = radius; x < level.width - radius; x += 2) {
      let solid = true;
      for (let dy = -radius; dy <= radius && solid; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (!diggableAt(terrain, x + dx, y + dy)) {
            solid = false;
            break;
          }
        }
      }
      if (solid) return { x, y };
    }
  }
  return null;
}

const solidPixels = (world, materialFlags) => {
  let count = 0;
  for (const index of world.level.data) if ((materialFlags[index] & 8) === 0) count++;
  return count;
};

test("the headless engine is the same file the live adapter reads", { skip }, async () => {
  const engine = await loadEngine();
  assert.equal(
    engine.sha256,
    CLIENT_SHA256,
    "the physics only transfer while this is the very same bundle",
  );
  assert.equal(engine.settings.name, "Liero 1.33");
  assert.equal(engine.weaponNames.length, 40);
  assert.equal(engine.materialFlags.length, 256);
});

test("a generated level comes back the same from the same seed", { skip }, async () => {
  const engine = await loadEngine();
  const digest = (level) => Buffer.from(level.data).toString("base64").slice(0, 64);
  const first = engine.randomLevel(7);
  assert.equal(digest(engine.randomLevel(7)), digest(first));
  assert.notEqual(digest(engine.randomLevel(8)), digest(first));
  assert.equal(first.width, 504);
  assert.equal(first.height, 350);
  // A mirrored map gives both worms the same ground to fight over.
  assert.equal(engine.randomLevel(7, { mirrored: true }).width, 1008);
  // The generator borrows Math.random for its terrain shape and has to give it
  // straight back, or every other random thing in the process is now seeded.
  const before = Math.random;
  engine.randomLevel(9);
  assert.equal(Math.random, before);
});

test("a generated level is dirt to dig, not rock to walk around", { skip }, async () => {
  const engine = await loadEngine();
  const census = (level) => {
    const totals = { air: 0, dirt: 0, rock: 0 };
    for (const index of level.data) {
      const flags = engine.materialFlags[index];
      if (flags & 8) totals.air++;
      else if (flags & 3) totals.dirt++;
      else totals.rock++;
    }
    return totals;
  };
  // Worth knowing before picking a training map: the stock .lev files in the kit
  // are almost entirely rock, so nothing an agent does there changes the ground.
  // A generated level is mostly dirt, which is the Liero the game is about.
  const generated = census(engine.randomLevel(3));
  assert.ok(generated.dirt > 50_000, `generated dirt ${generated.dirt}`);
  assert.ok(generated.dirt > generated.rock);
});

test("every key does what actions.js says it does", { skip }, async () => {
  const { world, worms } = await arena();
  const [worm, other] = worms;
  assert.equal(packKeys({ right: true, fire: true }), KEYS.right | KEYS.fire);
  assert.deepEqual(unpackKeys(KEYS.jump).jump, true);
  assert.equal(describeKeys(KEYS.left | KEYS.dig), "left+dig");
  assert.equal(ALL_KEYS, 511);

  // Left and right walk and turn the worm around.
  settle(world, worms);
  const start = worm.x;
  for (let tick = 0; tick < 30; tick++) {
    applyAction(world, worm, KEYS.right);
    other.Wa = 0;
    world.update();
  }
  assert.ok(worm.x > start, "right walks right");
  assert.equal(worm.direction, 1);
  const turned = worm.x;
  for (let tick = 0; tick < 30; tick++) {
    applyAction(world, worm, KEYS.left);
    other.Wa = 0;
    world.update();
  }
  assert.ok(worm.x < turned, "left walks left");
  assert.equal(worm.direction, 0);

  // Aim runs between the engine's own limits and stops there.
  const aim = worm.Oa;
  for (let tick = 0; tick < 60; tick++) {
    applyAction(world, worm, KEYS.aimUp);
    world.update();
  }
  assert.ok(worm.Oa > aim, "aimUp raises the aim");
  for (let tick = 0; tick < 240; tick++) {
    applyAction(world, worm, KEYS.aimDown);
    world.update();
  }
  assert.ok(worm.Oa < aim, "aimDown lowers it");
  assert.ok(worm.Oa >= world.s.pb.qh, "and it never passes the engine's limit");

  // Fire spends a round and puts something in the world.
  worm.Ka = 0;
  const ammo = worm.O[0].ha;
  applyAction(world, worm, KEYS.fire);
  world.update();
  assert.equal(worm.O[0].ha, ammo - 1, "one shotgun round");
  assert.ok(world.Ib.$ > 0, "and pellets in the air");
  assert.ok(worm.O[0].wd > 0, "with a cooldown before the next shot");
});

test("jump and dig fire on the press, not while the key is held", { skip }, async () => {
  const engine = await loadEngine();
  const { world, worms } = await arena({ seed: 12 });
  const [worm, other] = worms;
  settle(world, worms, 80);
  assert.ok(Math.abs(worm.b) < 0.01, "standing still on the ground");
  applyAction(world, worm, KEYS.jump);
  world.update();
  const launched = worm.b;
  assert.ok(launched < -0.5, `jump pushes the worm up, got ${launched}`);
  applyAction(world, worm, KEYS.jump);
  world.update();
  assert.ok(worm.b > launched, "holding the key does not jump again");
  assert.equal(EDGE_TRIGGERED, KEYS.jump | KEYS.dig);

  // Dig only removes dirt, so the worm has to be in some.
  const spot = findDirt(world, engine.materialFlags);
  assert.ok(spot, "a generated level should have a pocket of plain dirt");
  const hold = () => {
    worm.x = spot.x;
    worm.y = spot.y;
    worm.f = 0;
    worm.b = 0;
    worm.direction = 1;
    worm.Oa = 0;
    other.Wa = 0;
  };
  hold();
  applyAction(world, worm, 0); // released, so the next press is seen
  world.update();
  const before = solidPixels(world, engine.materialFlags);
  hold();
  applyAction(world, worm, KEYS.dig);
  world.update();
  const dug = before - solidPixels(world, engine.materialFlags);
  assert.ok(dug > 0, `dig removes dirt, removed ${dug}`);
  hold();
  applyAction(world, worm, KEYS.dig);
  world.update();
  assert.equal(
    solidPixels(world, engine.materialFlags),
    before - dug,
    "and holding it removes nothing more",
  );
});

test("the rope and the weapon are messages, not held keys", { skip }, async () => {
  const { world, worms } = await arena({ seed: 31 });
  const [worm] = worms;
  settle(world, worms);
  assert.equal(worm.Fa.Sc, false);
  applyAction(world, worm, { keys: 0, rope: ROPE.throw });
  assert.equal(worm.Fa.Sc, true, "the rope is out");
  world.update();
  applyAction(world, worm, { keys: 0, rope: ROPE.release });
  assert.equal(worm.Fa.Sc, false, "and let go");

  // The game's own weapon message is a relative move through the five slots.
  assert.equal(worm.Ka, 0);
  applyAction(world, worm, { keys: 0, weapon: 1 });
  assert.equal(worm.Ka, 1);
  applyAction(world, worm, { keys: 0, weapon: -1 });
  assert.equal(worm.Ka, 0);
  applyAction(world, worm, { keys: 0, weapon: -1 });
  assert.equal(worm.Ka, 4, "and it wraps, the way the key does");
  // The dig bit is not weapon change, however much its value suggests it.
  applyAction(world, worm, KEYS.dig);
  assert.equal(worm.Ka, 4);
});

test("a dead worm is dropped from the world and comes back whole", { skip }, async () => {
  const { world, worms } = await arena({ seed: 77 });
  const [worm] = worms;
  settle(world, worms);
  worm.Xa = 0;
  world.update();
  assert.equal(worm.u, false);
  assert.equal(world.za.includes(worm), false, "the engine drops it from za");
  respawnWorm(world, worm, LOADOUT);
  assert.equal(worm.Xa, 100);
  assert.equal(world.za.includes(worm), true);
  const start = worm.x;
  for (let tick = 0; tick < 40; tick++) {
    applyAction(world, worm, KEYS.right);
    world.update();
  }
  assert.notEqual(worm.x, start, "and it is simulated again");
});

test("an observation off a real map measures the real ground", { skip }, async () => {
  const engine = await loadEngine();
  const { world, worms } = await arena({ seed: 55 });
  settle(world, worms, 120);
  const view = viewFromWorld(world, worms[0], [worms[1]]);
  const vector = encodeVector(view);
  const patch = encodePatch(view);
  assert.equal(vector[VECTOR_OFFSETS.health], 1, "unhurt");
  assert.equal(vector[VECTOR_OFFSETS.foe], 1, "the other worm is alive");
  const down = vector[VECTOR_OFFSETS.rays + 4];
  assert.ok(down > 0 && down < 1, `the ground is within ray range, got ${down}`);
  const plane = PATCH.cells * PATCH.cells;
  const dirt = patch.subarray(plane, 2 * plane).reduce((sum, one) => sum + one, 0);
  assert.ok(dirt > 0, "a generated level puts dirt in the patch");
  const free = patch.subarray(2 * plane, 3 * plane);
  assert.equal(free[(PATCH.cells >> 1) * PATCH.cells + (PATCH.cells >> 1)], 1,
    "the worm itself stands in open space");
});

test("two rollouts of one seed are the same rollout", { skip }, async () => {
  const engine = await loadEngine();
  const rollout = (seed) => {
    const env = new WormEnv(engine, { episodeTicks: 400, seed });
    env.reset({ seed });
    let rng = 12345;
    const random = () => (rng = (Math.imul(1664525, rng) + 1013904223) >>> 0) / 2 ** 32;
    const trace = [];
    let done = false;
    while (!done) {
      const action = () =>
        (random() < 0.5 ? KEYS.left : KEYS.right) |
        (random() < 0.4 ? KEYS.aimUp : 0) |
        (random() < 0.3 ? KEYS.fire : 0) |
        (random() < 0.1 ? KEYS.jump : 0);
      const step = env.step([action(), action()]);
      done = step.done;
      trace.push(step.rewards.join(","));
    }
    return {
      trace: trace.join("|"),
      state: env.worms.map((worm) => [
        worm.x.toFixed(6),
        worm.y.toFixed(6),
        worm.Xa.toFixed(4),
      ]),
      totals: env.info().totals,
    };
  };
  const first = rollout(99);
  const again = rollout(99);
  const other = rollout(100);
  assert.deepEqual(again, first, "same seed, same episode down to the decimals");
  assert.notDeepEqual(other.state, first.state, "a different seed is a different fight");
});

test("a step is exactly frameskip ticks, and latency delays the worm", { skip }, async () => {
  const engine = await loadEngine();
  const walk = (inputLatencyTicks) => {
    const env = new WormEnv(engine, { frameskip: 4, inputLatencyTicks, seed: 5 });
    env.reset({ seed: 5 });
    assert.deepEqual(env.info().inputLatencyTicks, [inputLatencyTicks, inputLatencyTicks]);
    // Stand still first, so the comparison is about walking and not falling.
    for (let step = 0; step < 20; step++) env.step([0, 0]);
    const before = env.worms[0].x;
    const tick = env.world.qb;
    env.step([KEYS.right, 0]);
    return { moved: env.worms[0].x - before, ticks: env.world.qb - tick };
  };
  const prompt = walk(0);
  const delayed = walk(4);
  assert.equal(prompt.ticks, 4, "one step is frameskip ticks, no more");
  assert.ok(prompt.moved > 0, "with no latency the worm moves within the step");
  assert.equal(delayed.moved, 0, "a whole step of latency means it has not moved yet");
});

test("the reward follows the fight, for both sides at once", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, { seed: 8, episodeTicks: 6000 });
  env.reset({ seed: 8 });
  for (let step = 0; step < 20; step++) env.step([0, 0]);
  // Stand the two worms next to each other and let one of them fire.
  const [shooter, target] = env.worms;
  target.x = shooter.x + 12;
  target.y = shooter.y;
  shooter.direction = 1;
  shooter.Oa = 0;
  shooter.Ka = 0;
  let dealt = 0;
  let taken = 0;
  for (let step = 0; step < 12; step++) {
    const out = env.step([KEYS.fire, 0]);
    dealt += out.info.events[0].damageDealt;
    taken += out.info.events[0].damageTaken;
  }
  assert.ok(dealt > 0, `the shooter is paid for the damage, got ${dealt}`);
  const totals = env.info().totals;
  assert.equal(
    totals[1].damageTaken,
    totals[0].damageDealt,
    "one worm's damage dealt is the other's damage taken",
  );
  assert.ok(taken <= dealt, "and being shot at point blank costs the target more");
});

test("an episode ends on its tick budget and reset starts a clean one", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, { frameskip: 4, episodeTicks: 40, seed: 3 });
  env.reset({ seed: 3 });
  let steps = 0;
  let done = false;
  while (!done) {
    done = env.step([0, 0]).done;
    steps++;
  }
  assert.equal(steps, 10, "40 ticks at 4 ticks a step");
  assert.throws(() => env.step([0, 0]), /episode is over/);
  const { info } = env.reset();
  assert.equal(info.episode, 2);
  assert.equal(info.elapsedTicks, 0);
  assert.notEqual(info.seed, 3, "an unseeded reset moves on to the next episode");
  assert.equal(env.worms.length, 2);
  assert.equal(env.worms[0].Xa, 100);
});
