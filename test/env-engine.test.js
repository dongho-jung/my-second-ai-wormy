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
  DEFAULT_MOD,
  MODS_DIR,
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
import { WormEnv, directFire } from "../src/env/env.js";
import {
  PATCH,
  VECTOR_OFFSETS,
  encodePatch,
  encodeVector,
} from "../src/env/observation.js";
import { viewFromWorld } from "../src/env/view.js";
import { diggableAt } from "../src/env/terrain.js";

const absent = Object.values(ASSETS).filter(
  (name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)),
);
const skip = absent.length
  ? `no engine kit in ${DEFAULT_ENGINE_DIR.pathname}: missing ${absent.join(", ")} — run artifacts/headless-sim/fetch-assets.sh`
  : !existsSync(new URL(`${DEFAULT_MOD}/mod.json5`, MODS_DIR))
    ? `no ${DEFAULT_MOD} in ${MODS_DIR.pathname} — run: npm run mods`
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
  // The mod the watched room is set to. Not the stock one, and not the same
  // weapons: a policy is trained on whichever game it will actually meet.
  assert.equal(engine.settings.name, "csliero rewormed v0.37");
  assert.equal(engine.weaponNames.length, 129);
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

test("a jump press lets go of the rope, and letting go is a jump press", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    inputLatencyTicks: 0,
    seed: 5,
  });
  env.reset({ seed: 5 });
  const worm = env.worms[0];
  const step = (action) => env.step([action]);
  // Straight up, so the rope has a ceiling to find on any map.
  worm.Oa = Math.PI / 2;
  worm.ub = 0;
  step({ keys: 0, rope: ROPE.throw });
  assert.ok(env.views[0].self.rope, "the rope is out");
  // A press of Jump while the rope is out is the game's release, and the
  // environment says so in the decision it records.
  step({ keys: KEYS.jump });
  assert.equal(env.views[0].self.rope, null, "a jump press lets go");
  assert.equal(env.lastActions[0].rope, ROPE.release);
  // Held across two decisions it was pressed once: the second is no release.
  step({ keys: 0, rope: ROPE.throw });
  assert.ok(env.views[0].self.rope);
  step({ keys: KEYS.jump });
  assert.equal(env.views[0].self.rope, null);
  step({ keys: 0, rope: ROPE.throw });
  assert.ok(env.views[0].self.rope);
  step({ keys: KEYS.jump });
  assert.equal(env.views[0].self.rope, null, "pressed again, released again");
  step({ keys: 0, rope: ROPE.throw });
  step({ keys: KEYS.jump }); // released
  step({ keys: 0, rope: ROPE.throw });
  assert.ok(env.views[0].self.rope);
  step({ keys: KEYS.jump, rope: ROPE.throw }); // a throw with the press wins, as it does in the client
  assert.ok(env.views[0].self.rope, "throwing on the same decision keeps the rope");
  step({ keys: KEYS.jump }); // still held: no new press, so no release
  assert.ok(env.views[0].self.rope, "a key still held is not a press");
  // And the policy's own release is a Jump press in the keys it is recorded as.
  step({ keys: 0, rope: ROPE.release });
  assert.equal(env.views[0].self.rope, null);
  assert.equal(env.lastActions[0].keys & KEYS.jump, KEYS.jump, "release presses Jump");
  assert.equal(env.lastActions[0].rope, ROPE.release);
});

test("a held throw is not let go of, by a jump or another throw, until the hold is over", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    inputLatencyTicks: 0,
    ropeHold: 4,
    seed: 5,
  });
  env.reset({ seed: 5 });
  const worm = env.worms[0];
  const step = (action) => env.step([action]);
  worm.Oa = Math.PI / 2;
  worm.ub = 0;
  step({ keys: 0, rope: ROPE.throw });
  assert.ok(env.views[0].self.rope, "the rope is out");
  assert.equal(env.views[0].ropeHoldShare, 1, "and the hold has just started");
  const anchor = { ...env.views[0].self.rope.position };
  // Four decisions of jumping and throwing change nothing: the rope stays, on
  // the same anchor, and the recorded decisions carry neither.
  for (let decision = 0; decision < 4; decision++) {
    step({ keys: KEYS.jump, rope: ROPE.throw });
    assert.ok(env.views[0].self.rope, `still out after ${decision + 1}`);
    assert.equal(env.lastActions[0].rope, ROPE.none);
    assert.equal(env.lastActions[0].keys & KEYS.jump, 0, "the jump key is dropped");
  }
  assert.ok(env.views[0].self.rope.attached, "and it has hooked something by now");
  assert.deepEqual(env.views[0].self.rope.position, anchor, "on the anchor it first found");
  assert.equal(env.views[0].ropeHoldShare, 0, "the hold is over");
  // Now a jump press lets go, as it always did.
  step({ keys: KEYS.jump });
  assert.equal(env.views[0].self.rope, null);
  assert.equal(env.lastActions[0].rope, ROPE.release);
});

test("the rope flies through a wall the worm cannot", { skip }, async () => {
  // A corridor with a ceiling drawn in a colour that has no material flags:
  // the engine stops the worm at it and lets the rope through. Five of the
  // community maps draw much of their walls this way.
  const engine = await loadEngine();
  const flags = engine.materialFlags;
  const first = (want) => [...flags].findIndex((one) => one === want);
  const background = first(8);
  const rock = first(4);
  const ghost = first(0);
  assert.ok(background >= 0 && rock >= 0 && ghost >= 0, "the mod has all three");
  const level = engine.randomLevel(7);
  const { width: W, height: H } = level;
  const world = engine.createWorld({ level, seed: 1, rules: { bonusDrops: 0 } });
  const worm = engine.spawnWorm(world, { color: 0, playerId: 0, loadout: LOADOUT });
  const tick = (action, n) => {
    for (let i = 0; i < n; i++) {
      applyAction(world, worm, action);
      world.update();
    }
  };
  const results = {};
  for (const [name, wall] of [["rock", rock], ["ghost", ghost]]) {
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        level.data[y * W + x] = y >= 100 && y < 200 && x < 300 ? background : wall;
    // Of(), not createWorld: the world's level is a cached copy, and only Of()
    // writes into it.
    world.level.Of(level);
    worm.u = true;
    worm.x = 250;
    worm.y = 190;
    worm.f = worm.b = 0;
    worm.direction = 1;
    worm.Oa = worm.ub = 0;
    worm.Fa.Sc = worm.Fa.jc = false;
    tick(0, 40);
    const floor = worm.y;
    tick({ keys: KEYS.right }, 200);
    const walked = worm.x;
    worm.x = 150;
    worm.y = 190;
    worm.f = worm.b = 0;
    worm.Oa = Math.PI / 2;
    tick(0, 20);
    tick({ keys: 0, rope: ROPE.throw }, 1);
    tick(0, 40);
    results[name] = { floor, walked, attached: worm.Fa.jc, ropeY: worm.Fa.y };
  }
  for (const name of ["rock", "ghost"]) {
    assert.ok(Math.abs(results[name].floor - 196) < 1, `${name}: the worm stands on it`);
    assert.ok(results[name].walked >= 295 && results[name].walked < 300, `${name}: the wall stops the worm`);
  }
  assert.equal(results.rock.attached, true, "the rope holds on rock");
  assert.ok(results.rock.ropeY >= 95 && results.rock.ropeY <= 100, "at the ceiling");
  assert.ok(results.ghost.ropeY < 60, "through the ghost ceiling and on");
});

test("goals move out as the world's decisions add up", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    goals: "random",
    goalRadiusPx: [100, 500],
    goalRadiusFullAt: 100,
    goalPatience: 10,
    seed: 3,
  });
  env.reset();
  assert.equal(env.goalRadius(), 100);
  const worm = env.views[0].self.position;
  const goal = env.progress[0].goal;
  assert.ok(goal, "a goal was handed out");
  assert.ok(Math.hypot(goal.x - worm.x, goal.y - worm.y) <= 100, "and it is within the radius");
  for (let decision = 0; decision < 50; decision++) env.step([0]);
  assert.equal(env.goalRadius(), 300, "half way there half way through");
  for (let decision = 0; decision < 60; decision++) env.step([0]);
  assert.equal(env.goalRadius(), 500, "and it stops at the far end");
  assert.equal(env.info().goalRadiusPx, 500);
  // Standing still for 110 decisions at a patience of 10 gave up on ten goals
  // or so, each swapped for another rather than kept.
  assert.ok((env.totals[0].goalsMissed ?? 0) >= 9, `gave up on ${env.totals[0].goalsMissed} goals`);
  assert.ok(env.progress[0].goal, "and there is still one to go to");
  // A world carried on from a checkpoint starts where the radius had got to.
  const carried = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    goals: "random",
    goalRadiusPx: [100, 500],
    goalRadiusFullAt: 100,
    decisionsDone: 75,
    seed: 3,
  });
  assert.equal(carried.goalRadius(), 400);
});

test("a world that keeps giving up on its goals draws them closer", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    goals: "random",
    goalRadiusPx: [96, 1600],
    goalRadiusMode: "success",
    goalRadiusStart: 300,
    goalCurriculum: { window: 5 },
    goalPatience: 10,
    seed: 3,
  });
  env.reset();
  assert.equal(env.goalRadius(), 300, "starts where it was told");
  // A worm that presses nothing gives up on a goal every ten decisions, and
  // every five of those is a window of nothing reached: a notch back each.
  for (let decision = 0; decision < 100; decision++) env.step([0]);
  const missed = env.totals[0].goalsMissed ?? 0;
  assert.ok(missed >= 9, `gave up on ${missed} goals`);
  const notches = Math.floor(missed / 5);
  assert.ok(Math.abs(env.goalRadius() - 300 / 1.1 ** notches) < 1e-6, `${notches} notches back`);
  assert.equal(env.info().goalRadiusPx, env.goalRadius());
  // The clock does nothing to it.
  assert.ok(env.goalRadius() < 300, "moved, and not by the clock");
});

test("reliable arrivals tighten a world's goal deadline", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 1,
    observations: ["vector"],
    goals: "random",
    goalRadiusPx: 100,
    goalPatience: 10,
    goalPatienceMin: 5,
    goalDeadlineCurriculum: { window: 2 },
    seed: 4,
  });
  env.reset();
  for (let arrival = 0; arrival < 2; arrival++) {
    const { position } = env.views[0].self;
    env.progress[0].setGoal({ ...position }, position);
    env.step([0]);
  }
  assert.ok(Math.abs(env.goalDeadline() - 10 / 1.1) < 1e-9);
  assert.equal(env.totals[0].goalsReached, 2);
  assert.ok(env.totals[0].goalStepsReached >= 2, "arrival time is retained for the episode stats");
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
  assert.equal(vector[VECTOR_OFFSETS.foes], 1, "the other worm is alive");
  const down = vector[VECTOR_OFFSETS.rays + 4];
  assert.ok(down > 0 && down < 1, `the ground is within ray range, got ${down}`);
  const plane = PATCH.columns * PATCH.rows;
  const dirt = patch.subarray(plane, 2 * plane).reduce((sum, one) => sum + one, 0);
  assert.ok(dirt > 0, "a generated level puts dirt in the patch");
  const free = patch.subarray(2 * plane, 3 * plane);
  assert.equal(free[(PATCH.rows >> 1) * PATCH.columns + (PATCH.columns >> 1)], 1,
    "the worm itself stands in open space");
});

test("two rollouts of one seed are the same rollout", { skip }, async () => {
  const engine = await loadEngine();
  const rollout = (seed) => {
    const env = new WormEnv(engine, { agents: 3, episodeTicks: 400, seed });
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
      const step = env.step([action(), action(), action()]);
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
    const env = new WormEnv(engine, {
      agents: 2,
      frameskip: 4,
      inputLatencyTicks,
      seed: 5,
    });
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

test("the reward goes to whoever earned it, with a third worm watching", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, {
    agents: 3,
    seed: 8,
    episodeTicks: 6000,
    // A shotgun each, so the test is not at the mercy of a random loadout.
    loadout: [0, 0, 0, 0, 0],
  });
  env.reset({ seed: 8 });
  for (let step = 0; step < 20; step++) env.step([0, 0, 0]);
  // Stand two of them nose to nose and leave the third where it is.
  const [shooter, target, bystander] = env.worms;
  target.x = shooter.x + 12;
  target.y = shooter.y;
  shooter.direction = 1;
  shooter.Oa = 0;
  shooter.Ka = 0;
  let dealt = 0;
  let bystanderReward = 0;
  for (let step = 0; step < 12; step++) {
    const out = env.step([KEYS.fire, 0, 0]);
    dealt += out.info.events[0].damageDealt;
    bystanderReward += out.rewards[2];
  }
  assert.ok(dealt > 0, `the shooter is paid for the damage, got ${dealt}`);
  const totals = env.info().totals;
  const sum = (field) => totals.reduce((all, one) => all + (one[field] ?? 0), 0);
  // Every point of health leaves one worm and is charged to exactly one worm,
  // and a hit on yourself is taken without being dealt. A shotgun spreads, so
  // some of it lands on the worm that was only watching — which is the whole
  // reason the engine's own attribution is worth reading instead of guessing
  // from who lost health.
  assert.equal(
    +(sum("damageTaken") - sum("selfDamage")).toFixed(6),
    +sum("damageDealt").toFixed(6),
  );
  assert.ok(
    totals[1].damageTaken - totals[1].selfDamage > 0,
    "the worm in front of the barrel took most of it",
  );
  assert.equal(totals[2].damageDealt, 0, "the third worm hit nobody");
  assert.ok(
    bystanderReward <= 0.5,
    `and is paid nothing for the fight it watched, got ${bystanderReward}`,
  );
  assert.ok(bystander.Xa > 0);
});

test("weapons are drawn fresh, from the pool the run asked for", { skip }, async () => {
  const engine = await loadEngine();
  const explodes = (id) => engine.settings.O[id].be?.Bd === 0;
  const guns = directFire(engine);
  // Plenty of weapons do their damage by exploding, and a policy that cannot
  // aim yet fires those at its own feet. Where the guns can be named, the
  // default pool has none of the rest.
  assert.ok(guns.every((id) => !explodes(id)), "the direct-fire pool must not explode");
  assert.ok(
    engine.settings.O.some((_, id) => explodes(id)),
    "and the mod must have some that do, or this test proves nothing",
  );

  const env = new WormEnv(engine, { agents: 3, episodeTicks: 120, seed: 2 });
  // A community mod names its weapons after real guns and matches none of the
  // list, and then the honest pool is every weapon rather than an empty one.
  const named = guns.length >= 5;
  assert.equal(env.weaponPool === null, !named, "an unnameable pool falls back to all of them");
  const seen = new Set();
  for (let episode = 0; episode < 6; episode++) {
    env.reset();
    for (const loadout of env.loadouts) {
      assert.equal(new Set(loadout).size, 5, "five different weapons");
      for (const id of loadout) {
        if (named) {
          assert.ok(guns.includes(id), `${engine.weaponNames[id]} is not direct fire`);
        }
        seen.add(id);
      }
    }
  }
  assert.ok(seen.size > 5, `six episodes should show more than one loadout, saw ${seen.size}`);

  const anything = new WormEnv(engine, { agents: 2, weaponPool: "all", episodeTicks: 120 });
  const drawn = new Set();
  for (let episode = 0; episode < 20; episode++) {
    anything.reset();
    for (const loadout of anything.loadouts) for (const id of loadout) drawn.add(id);
  }
  assert.ok([...drawn].some(explodes), "`all` has to reach the explosives");
  assert.throws(() => new WormEnv(engine, { weaponPool: "nope" }), /unknown weapon pool/);
});

test("any number of worms, from a solo run to a brawl", { skip }, async () => {
  const engine = await loadEngine();
  for (const agents of [1, 2, 5]) {
    const env = new WormEnv(engine, { agents, episodeTicks: 240, seed: 3 });
    const { observations } = env.reset({ seed: 3 });
    // The vector describes every other worm, so its length follows the count.
    assert.equal(env.spec.foeSlots, agents - 1);
    assert.equal(observations.length, agents);
    assert.equal(observations[0].vector.length, env.spec.vectorSize);
    assert.equal(env.worms.length, agents);
    let done = false;
    while (!done) done = env.step(new Array(agents).fill(KEYS.right)).done;
    assert.equal(env.info().totals.length, agents);
    assert.throws(() => env.reset() && env.step(new Array(agents + 1).fill(0)), /expected/);
  }
  // Sizing the vector for more foes than are playing lets one policy do both:
  // the empty slots are zeros, not a different shape.
  const duel = new WormEnv(engine, { agents: 2, observationFoes: 4, episodeTicks: 120 });
  const brawl = new WormEnv(engine, { agents: 5, observationFoes: 4, episodeTicks: 120 });
  assert.equal(duel.spec.vectorSize, brawl.spec.vectorSize);
  duel.reset({ seed: 1 });
  assert.equal(duel.observations[0].vector.length, brawl.spec.vectorSize);
  assert.throws(() => new WormEnv(engine, { agents: 0 }), /at least 1/);
});

test("an episode ends on its tick budget and reset starts a clean one", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, { frameskip: 4, episodeTicks: 40, seed: 3 });
  env.reset({ seed: 3 });
  let steps = 0;
  let done = false;
  const idle = new Array(env.agents).fill(0);
  while (!done) {
    done = env.step(idle).done;
    steps++;
  }
  assert.equal(steps, 10, "40 ticks at 4 ticks a step");
  assert.throws(() => env.step(idle), /episode is over/);
  const { info } = env.reset();
  assert.equal(info.episode, 2);
  assert.equal(info.elapsedTicks, 0);
  assert.notEqual(info.seed, 3, "an unseeded reset moves on to the next episode");
  assert.equal(env.worms.length, env.agents);
  assert.equal(env.worms[0].Xa, 100);
});

test("the observation carries the decision that came before it", { skip }, async () => {
  const engine = await loadEngine();
  const env = new WormEnv(engine, { agents: 2, episodeTicks: 400, seed: 5 });
  const { observations } = env.reset({ seed: 5 });
  const at = env.spec.offsets.lastAction;
  const width = env.spec.offsets.latency - at;
  const block = (observation) => Array.from(observation.vector.subarray(at, at + width));
  assert.deepEqual(block(observations[0]), new Array(width).fill(0), "nothing chosen yet");
  const step = env.step([{ keys: KEYS.right | KEYS.fire, rope: 1, weapon: -1 }, KEYS.left]);
  assert.deepEqual(block(step.observations[0]), [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, -1]);
  assert.deepEqual(block(step.observations[1]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  // The next match starts with nothing chosen again.
  env.reset({ seed: 6 });
  assert.deepEqual(block(env.observations[0]), new Array(width).fill(0));
});

test("a world told how far the ladder had faded carries on from there", { skip }, async () => {
  const engine = await loadEngine();
  const resumed = new WormEnv(engine, {
    agents: 2, episodeTicks: 400, seed: 3, shapingFullAt: 100, decisionsDone: 50,
  });
  assert.equal(resumed.shaping, 0.5, "half the fade was already done");
  resumed.reset({ seed: 3 });
  resumed.step([0, 0]);
  assert.ok(Math.abs(resumed.shaping - 0.49) < 1e-9, "and it goes on from there");
  const fresh = new WormEnv(engine, { agents: 2, episodeTicks: 400, seed: 3, shapingFullAt: 100 });
  assert.equal(fresh.shaping, 1, "a new run starts at the top");

  const speedPhase = new WormEnv(engine, {
    agents: 2,
    episodeTicks: 400,
    seed: 3,
    decisionsDone: 500,
    goalProgressStartAt: 500,
    goalProgressFullAt: 100,
    goalProgressFloor: 0.1,
  });
  assert.equal(speedPhase.goalProgressScale, 1, "a resumed speed phase starts a new fade now");
  speedPhase.reset({ seed: 3 });
  speedPhase.step([0, 0]);
  assert.ok(
    Math.abs(speedPhase.goalProgressScale - 0.991) < 1e-9,
    "only the new phase's decisions advance it",
  );
});
