// A movement Watch overlays several attempts, but every attempt must remain a
// complete world of its own. Sharing even the terrain buffer would let one
// ghost dig a path for another and turn the picture back into a match.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { basename } from "node:path";
import { KEYS } from "../src/env/actions.js";
import { ASSETS, DEFAULT_ENGINE_DIR, DEFAULT_MOD, MODS_DIR, loadEngine } from "../src/env/engine.js";
import { WormEnv } from "../src/env/env.js";
import { GhostRace } from "../src/env/race.js";

const levelPath = new URL("../artifacts/maps/dsds-cs/cs_ash3_remix.png", import.meta.url);
const absent = Object.values(ASSETS).filter(
  (name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)),
);
const skip = absent.length
  ? `no engine kit in ${DEFAULT_ENGINE_DIR.pathname}`
  : !existsSync(new URL(`${DEFAULT_MOD}/mod.json5`, MODS_DIR))
    ? `no ${DEFAULT_MOD} in ${MODS_DIR.pathname}`
    : !existsSync(levelPath)
      ? `no test map at ${levelPath.pathname}`
      : false;

test("a fixed race gives every ghost exact endpoints and no shared world state", { skip }, async () => {
  const engine = await loadEngine();
  const file = basename(levelPath.pathname);
  const level = engine.readAnyLevel(file, readFileSync(levelPath));

  // Let the environment find one valid spawn/goal pair on this real map. The
  // race then has to replay those recorded coordinates exactly.
  const probe = new WormEnv(engine, {
    agents: 1,
    level,
    goals: "random",
    goalRadiusPx: 1600,
    goalsPerEpisode: 1,
    observations: ["vector"],
    seed: 913,
  });
  probe.reset({ seed: 913 });
  const start = { ...probe.views[0].self.position };
  const goal = { ...probe.progress[0].goal };
  assert.ok(goal, "the real map should provide a movement goal");

  const levels = new Map([
    [file, level],
    [level.name, level],
  ]);
  const race = new GhostRace(engine, {
    racers: 3,
    scenarios: [{ seed: 913, map: file, start: [start.x, start.y], goal: [goal.x, goal.y] }],
    levels,
    world: {
      observations: ["vector", "patchBytes", "map"],
      lockWeapons: true,
      loadout: [0, 2, 3, 5, 10],
    },
    episodeTicks: 120,
  });

  assert.equal(race.envs.length, 3);
  assert.equal(new Set(race.envs.map((env) => env.world)).size, 3, "worlds are independent");
  assert.equal(
    new Set(race.envs.map((env) => env.world.level.data.buffer)).size,
    3,
    "terrain buffers are independent",
  );
  for (const env of race.envs) {
    assert.deepEqual([env.worms[0].x, env.worms[0].y], [start.x, start.y]);
    assert.deepEqual([env.progress[0].goal.x, env.progress[0].goal.y], [goal.x, goal.y]);
  }

  const otherX = race.envs[1].worms[0].x;
  race.envs[0].worms[0].x += 25;
  assert.equal(race.envs[1].worms[0].x, otherX, "moving one ghost cannot move another");

  const terrainAt = race.envs[1].world.level.data[0];
  race.envs[0].world.level.data[0] ^= 1;
  assert.equal(race.envs[1].world.level.data[0], terrainAt, "digging one world cannot alter another");

  race.step([KEYS.right, KEYS.left, 0]);
  assert.equal(race.envs[0].lastActions[0].keys, KEYS.right);
  assert.equal(race.envs[1].lastActions[0].keys, KEYS.left);
  assert.equal(race.envs[2].lastActions[0].keys, 0);
});
