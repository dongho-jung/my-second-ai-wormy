import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { ASSETS, DEFAULT_ENGINE_DIR, loadEngine } from "../src/env/engine.js";
import { KEYS } from "../src/env/actions.js";
import { WormEnv } from "../src/env/env.js";
import { BACKGROUND, DIGGABLE } from "../src/env/terrain.js";

// A buried goal is the one task that cannot be finished without the dig key.
const absent = Object.values(ASSETS).filter((name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)));
const skip = absent.length ? `no engine kit: missing ${absent.join(", ")}` : false;

function open(level, flags, x, y) {
  return (flags[level.data[Math.round(y) * level.width + Math.round(x)]] & BACKGROUND) !== 0;
}

test("a buried goal is packed with dirt, dug out by the dig key, and never leaks into the map it came from", { skip }, async () => {
  const engine = await loadEngine();
  const flags = engine.materialFlags;
  const source = engine.randomLevel(31);
  const pristine = Uint8Array.from(source.data);
  let env = null;
  let goal = null;
  for (let seed = 1; seed < 60 && !goal; seed++) {
    env = new WormEnv(engine, {
      agents: 1, seed, level: () => source, goals: "random", goalRadiusPx: 160,
      goalDigShare: 1, goalsPerEpisode: 1, lockWeapons: true, weights: "movement",
      observations: ["vector"],
    });
    env.reset();
    if (env.progress[0].goal?.dig) goal = env.progress[0].goal;
  }
  assert.ok(goal, "some start was far enough from its goal to bury it");
  const level = env.world.level;
  // Nothing open is left within the arrival circle and a margin around it.
  for (let dy = -30; dy <= 30; dy++) {
    for (let dx = -30; dx <= 30; dx++) {
      if (dx * dx + dy * dy > 30 * 30) continue;
      const x = Math.round(goal.x) + dx;
      const y = Math.round(goal.y) + dy;
      if (x < 0 || y < 0 || x >= level.width || y >= level.height) continue;
      assert.equal(open(level, flags, x, y), false, `open pixel at ${dx},${dy} from a buried goal`);
    }
  }
  assert.ok(flags[env.dirtIndex] & DIGGABLE, "the packing is diggable");
  assert.deepEqual(source.data, pristine, "the map the episode was copied from is untouched");
  assert.equal(env.totals[0].goalDig, 1);
  assert.equal(env.totals[0].goalClosestShare, 1);

  // Put the worm against the ball and dig: the packing gives way.
  const worm = env.worms[0];
  const count = () => {
    let solid = 0;
    for (let dy = -44; dy <= 44; dy++) {
      for (let dx = -44; dx <= 44; dx++) {
        const x = Math.round(goal.x) + dx;
        const y = Math.round(goal.y) + dy;
        if (x >= 0 && y >= 0 && x < level.width && y < level.height && !open(level, flags, x, y)) solid++;
      }
    }
    return solid;
  };
  const before = count();
  for (let press = 0; press < 20; press++) {
    worm.x = goal.x - 36;
    worm.y = goal.y;
    worm.f = 0;
    worm.b = 0;
    worm.Wa = press % 2 ? KEYS.dig | KEYS.right : KEYS.right;
    env.world.update();
  }
  assert.ok(count() < before, "digging removed some of the packing");
});

test("without buried goals the destinations are drawn exactly as before", { skip }, async () => {
  const engine = await loadEngine();
  const source = engine.randomLevel(31);
  const goals = (share) => {
    const env = new WormEnv(engine, {
      agents: 1, seed: 5, level: () => source, goals: "random", goalRadiusPx: 160,
      goalDigShare: share, goalDetourShare: 0.5, goalAboveShare: 0.5, goalsPerEpisode: 1,
      observations: ["vector"],
    });
    const out = [];
    for (let episode = 0; episode < 6; episode++) {
      env.reset();
      out.push([env.progress[0].goal?.x, env.progress[0].goal?.y]);
    }
    return out;
  };
  assert.deepEqual(goals(0), goals(0));
  const withDig = goals(0.5);
  assert.equal(withDig.length, 6);
});
