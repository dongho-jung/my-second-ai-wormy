// The batched environment and the wire it talks over.
//
// These need the game bundle, so they skip with a reason when the kit is not
// there — the same as env-engine.test.js.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  ACTION_HEADS,
  ACTION_SIZES,
  KEYS,
  ROPE,
  actionFromHeads,
} from "../src/env/actions.js";
import { ASSETS, DEFAULT_ENGINE_DIR, loadEngine } from "../src/env/engine.js";
import { DONE, EPISODE_STATS, HEADS, VecWormEnv } from "../src/env/vec.js";
import { PATCH_CELLS } from "../src/env/observation.js";

const absent = Object.values(ASSETS).filter(
  (name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)),
);
const skip = absent.length
  ? `no engine kit in ${DEFAULT_ENGINE_DIR.pathname}: missing ${absent.join(", ")} — run artifacts/headless-sim/fetch-assets.sh`
  : false;

test("the heads are the keys a person would press, factorised", () => {
  assert.equal(HEADS, 8);
  assert.deepEqual(ACTION_SIZES, [3, 3, 2, 2, 2, 2, 3, 3]);
  // Left and right together is the engine doing nothing, so they share a head
  // rather than being two bits that can contradict each other.
  assert.deepEqual(actionFromHeads([1, 0, 0, 0, 0, 0, 0, 0]), {
    keys: KEYS.left,
    rope: ROPE.none,
    weapon: 0,
  });
  assert.deepEqual(actionFromHeads([2, 2, 1, 1, 1, 1, 0, 2]), {
    keys: KEYS.right | KEYS.aimDown | KEYS.fire | KEYS.jump | KEYS.dig,
    rope: ROPE.throw,
    weapon: -1,
  });
  // There is no release choice: letting go is the jump head, as it is a
  // person's Jump key. A stray 2 on the rope head is read as nothing.
  assert.equal(actionFromHeads([0, 0, 0, 0, 0, 2, 0, 0]).rope, ROPE.none);
  // Reeling the rope in and paying it out are held keys, like moving and
  // aiming, and share a head for the same reason: both at once is nothing.
  // Throwing the rope is a separate message and can happen in the same tick.
  assert.deepEqual(actionFromHeads([0, 0, 0, 0, 0, 1, 1, 0]), {
    keys: KEYS.ropeShorter,
    rope: ROPE.throw,
    weapon: 0,
  });
  assert.deepEqual(actionFromHeads([0, 0, 0, 0, 0, 0, 2, 0]), {
    keys: KEYS.ropeLonger,
    rope: ROPE.none,
    weapon: 0,
  });
  // Read out of the middle of a batch, which is how they arrive.
  const batch = Uint8Array.from([
    0, 0, 0, 0, 0, 0, 0, 0,
    2, 1, 1, 0, 0, 1, 2, 1,
  ]);
  assert.equal(
    actionFromHeads(batch, 8).keys,
    KEYS.right | KEYS.aimUp | KEYS.fire | KEYS.ropeLonger,
  );
  assert.equal(actionFromHeads(batch, 8).rope, ROPE.throw);
  assert.equal(actionFromHeads(batch, 8).weapon, 1);
  assert.equal(
    ACTION_HEADS.reduce((all, [, choices]) => all * choices.length, 1),
    1296,
    "1,296 combinations, described by 20 numbers",
  );
});

test("a row of worlds writes into one buffer per kind", { skip }, async () => {
  const engine = await loadEngine();
  const vec = new VecWormEnv(engine, {
    envs: 4,
    agents: 3,
    episodeTicks: 200,
    levelPool: 2,
    seed: 5,
  });
  const slots = 4 * 3;
  assert.equal(vec.vectors.length, slots * vec.spec.vectorSize);
  assert.equal(vec.patches.length, slots * PATCH_CELLS);
  assert.equal(vec.rewards.length, slots);
  assert.equal(vec.dones.length, 4, "a match ends for all of its worms at once");
  vec.reset();

  // Each world writes its observation straight into the shared buffer, so no
  // two of them can be looking at the same numbers.
  const firsts = new Set();
  for (let slot = 0; slot < slots; slot++) {
    firsts.add(vec.vectors.subarray(slot * vec.spec.vectorSize, slot * vec.spec.vectorSize + 16).join());
  }
  assert.ok(firsts.size > 1, "different worlds, different observations");

  const heads = new Uint8Array(slots * HEADS);
  for (let at = 0; at < heads.length; at += HEADS) heads[at] = 2; // everyone right
  const before = vec.envs[0].worms[0].x;
  vec.step(heads);
  assert.ok(vec.envs[0].worms[0].x > before, "the heads reached the worms");
  assert.throws(() => vec.step(new Uint8Array(3)), /expected/);

  // Run past the end of an episode: the world restarts itself and says so.
  // An episode ends on a clock, not on anything the game did, so its last
  // observation is handed over to be valued and the world restarts on the next
  // call. That shows up as two bytes a step apart: `last`, then `first`.
  let closed = 0;
  let opened = 0;
  let heldAtClose = null;
  for (let step = 0; step < 60; step++) {
    vec.step(heads);
    for (const one of vec.dones) {
      if (one === DONE.last) closed++;
      if (one === DONE.first) opened++;
    }
    if (vec.dones[0] === DONE.last) {
      // Not reset yet: what was just sent is the end of the old episode.
      heldAtClose = vec.envs[0].done;
    }
  }
  assert.equal(closed, 4, "each of the four matches ended once");
  assert.equal(opened, 4, "and opened its next one a step later");
  assert.equal(heldAtClose, true, "the closing observation is sent before the reset");
  const stats = Object.fromEntries(
    EPISODE_STATS.map((field, index) => [field, vec.stats[index]]),
  );
  assert.equal(stats.steps, 50, "200 ticks at 4 ticks a step");
  assert.ok(Number.isFinite(stats.reward));
  assert.ok(stats.ropeShare >= 0 && stats.ropeShare <= 1, "a share of the match");
  assert.ok(stats.seed > 0);
  assert.equal(vec.envs[0].done, false, "and is already playing the next one");
});

test("one policy can drive any number of worms", { skip }, async () => {
  const engine = await loadEngine();
  let width = null;
  for (const agents of [1, 2, 5]) {
    const vec = new VecWormEnv(engine, {
      envs: 2,
      agents,
      observationFoes: 4,
      episodeTicks: 120,
      levelPool: 1,
    });
    vec.reset();
    // The vector is sized for four foes whatever is playing, so the same
    // network takes a solo run, a duel and a brawl.
    if (width === null) width = vec.describe().vectorSize;
    assert.equal(vec.describe().vectorSize, width);
    assert.equal(vec.vectors.length, 2 * agents * width);
    vec.step(new Uint8Array(2 * agents * HEADS));
  }
});

test("a resolved movement task is terminal and then opens a fresh task", { skip }, async () => {
  const engine = await loadEngine();
  const vec = new VecWormEnv(engine, {
    envs: 1,
    agents: 1,
    observations: ["vector"],
    goals: "random",
    goalsPerEpisode: 1,
    endOnGoals: true,
    goalRadiusPx: 100,
    episodeTicks: 3600,
    seed: 41,
  }).reset();
  const position = vec.envs[0].views[0].self.position;
  vec.envs[0].progress[0].setGoal({ ...position }, position);

  vec.step(new Uint8Array(HEADS));
  assert.equal(vec.dones[0], DONE.terminal);
  const stats = Object.fromEntries(
    EPISODE_STATS.map((field, index) => [field, vec.stats[index]]),
  );
  assert.equal(stats.goalsReached, 1);
  assert.equal(stats.goalsAssigned, 1);

  vec.step(new Uint8Array(HEADS));
  assert.equal(vec.dones[0], DONE.first);
  assert.equal(vec.envs[0].done, false);
});

test("a fixed exam cycles through maps instead of sampling an accidental subset", { skip }, async () => {
  const engine = await loadEngine();
  const vec = new VecWormEnv(engine, {
    envs: 2,
    agents: 1,
    observations: ["vector"],
    levelPool: 3,
    levelSequence: "roundRobin",
    levelStride: 2,
    seed: 19,
  }).reset();
  assert.deepEqual(vec.envs.map((env) => env.levelIndex), [0, 1]);

  vec.envs.forEach((env) => env.reset());
  assert.deepEqual(vec.envs.map((env) => env.levelIndex), [2, 0]);
  vec.envs.forEach((env) => env.reset());
  assert.deepEqual(vec.envs.map((env) => env.levelIndex), [1, 2]);
});

test("episode stats distinguish fast direct routes from merely reaching", () => {
  const fake = {
    opponents: 1,
    levels: [{}],
    stats: new Float32Array(EPISODE_STATS.length),
  };
  const env = {
    info: () => ({
      elapsedTicks: 3600,
      totals: [
        {
          goalsReached: 2,
          goalsMissed: 0,
          goalsAssigned: 2,
          goalAssignedPx: 640,
          goalStartX: 10,
          goalStartY: 20,
          goalTargetX: 300,
          goalTargetY: 400,
          goalStepsReached: 150,
          goalDirectPx: 600,
          goalPathPx: 750,
          fromGoalSpeed: 12,
        },
        {
          goalsReached: 1,
          goalsMissed: 1,
          goalsAssigned: 2,
          goalAssignedPx: 760,
          goalStartX: 30,
          goalStartY: 40,
          goalTargetX: 500,
          goalTargetY: 600,
          goalStepsReached: 90,
          goalDirectPx: 300,
          goalPathPx: 600,
          fromGoalSpeed: 4,
        },
      ],
    }),
    episodeTicks: 3600,
    frameskip: 4,
    shaping: 1,
    goalProgressScale: 0.5,
    goalRadius: () => 1600,
    goalDeadline: () => 300,
    episodeSeed: 7,
  };
  VecWormEnv.prototype.writeStats.call(fake, 0, env);
  const stats = Object.fromEntries(
    EPISODE_STATS.map((field, index) => [field, fake.stats[index]]),
  );
  assert.equal(stats.goalsReached, 1.5);
  assert.equal(stats.goalsAssigned, 2);
  assert.equal(stats.goalAssignedDistance, 350);
  assert.equal(stats.goalStartX, 20);
  assert.equal(stats.goalTargetY, 500);
  assert.equal(stats.goalSeconds, 5.5);
  assert.equal(stats.goalSpeed, 55);
  assert.ok(Math.abs(stats.goalPathEfficiency - 0.65) < 1e-6);
  assert.equal(stats.goalPatience, 300);
  assert.equal(stats.goalProgressScale, 0.5);
  assert.equal(stats.fromGoalSpeed, 8);
  assert.equal(stats.goalsVsPast, 1);
  assert.equal(stats.goalSecondsVsPast, -1);
  assert.equal(stats.goalSpeedVsPast, 10);
  assert.ok(Math.abs(stats.goalEfficiencyVsPast - 0.3) < 1e-6);
});

test("the worker speaks the frames it says it will", { skip }, async () => {
  const worker = fileURLToPath(new URL("../src/env/worker.js", import.meta.url));
  const config = JSON.stringify({
    envs: 2,
    agents: 3,
    episodeTicks: 120,
    levelPool: 1,
    seed: 9,
  });
  const child = spawn(process.execPath, [worker, config], {
    stdio: ["pipe", "pipe", "inherit"],
  });
  try {
    const frames = [];
    let waiting = null;
    let held = Buffer.alloc(0);
    child.stdout.on("data", (chunk) => {
      held = Buffer.concat([held, chunk]);
      for (;;) {
        if (held.length < 4) break;
        const size = held.readUInt32LE(0);
        if (held.length < 4 + size) break;
        const frame = held.subarray(4, 4 + size);
        held = held.subarray(4 + size);
        if (waiting) {
          const resolve = waiting;
          waiting = null;
          resolve(frame);
        } else frames.push(frame);
      }
    });
    const next = () =>
      frames.length
        ? Promise.resolve(frames.shift())
        : new Promise((resolve) => (waiting = resolve));

    const layout = JSON.parse((await next()).toString("utf8"));
    assert.equal(layout.envs, 2);
    assert.equal(layout.agents, 3);
    assert.equal(layout.patchCells, PATCH_CELLS);
    // What a byte expands to travels with the layout, so a trainer built for
    // another picture can refuse.
    assert.equal(layout.patchChannels, 8);
    assert.equal(layout.patchShape[0], 8);
    assert.equal(layout.mapChannels, 5);
    assert.equal(layout.mapCells, 5 * 32 * 32);
    assert.equal(layout.actionBytes, 2 * 3 * HEADS);
    assert.deepEqual(layout.order, ["vectors", "patches", "patches2", "maps", "rewards", "dones",
      "restarts", "stats"]);
    assert.equal(layout.patch2Cells, 0, "no second cut unless one was asked for");
    assert.equal(layout.bytes.patches2, 0);
    assert.ok(layout.mapCells > 0, "the whole level goes on the wire too");
    // The layout is the only thing a reader needs: every block's size is in it.
    const total = Object.values(layout.bytes).reduce((sum, one) => sum + one, 0);

    // The first observation arrives before anything has been asked for.
    const opening = await next();
    assert.equal(opening.length, total);

    const heads = Buffer.alloc(layout.actionBytes);
    for (let at = 0; at < heads.length; at += HEADS) heads[at] = 2;
    const size = Buffer.allocUnsafe(4);
    size.writeUInt32LE(heads.length, 0);
    child.stdin.write(Buffer.concat([size, heads]));
    const stepped = await next();
    assert.equal(stepped.length, total);
    // Read by the layout rather than by a remembered order: the frame grew a
    // whole-level block between the patches and the rewards.
    const offsets = {};
    let at = 0;
    for (const name of layout.order) {
      offsets[name] = at;
      at += layout.bytes[name];
    }
    const rewards = new Float32Array(
      stepped.buffer.slice(
        stepped.byteOffset + offsets.rewards,
        stepped.byteOffset + offsets.rewards + layout.bytes.rewards,
      ),
    );
    assert.equal(rewards.length, 6);
    assert.ok(rewards.every((value) => Number.isFinite(value)));
    assert.notDeepEqual(
      Buffer.compare(opening, stepped),
      0,
      "the world moved, so the frame is not the one before it",
    );
  } finally {
    child.kill();
  }
});

test("a match can cut the ground at two scales, one per side", { skip }, async () => {
  const engine = await loadEngine();
  const vec = new VecWormEnv(engine, { envs: 1, agents: 2, patchScale: 4, patchScale2: 2 });
  const layout = vec.describe();
  assert.deepEqual(layout.patchShape, [8, 61, 107]);
  assert.deepEqual(layout.patch2Shape, [8, 121, 213]);
  assert.equal(layout.patch2Scale, 2);
  assert.equal(vec.patches2.length, 2 * 121 * 213);
  const heads = new Uint8Array(2 * HEADS);
  vec.step(heads);
  // Both cuts show the same ground: a cell of the coarse one is rock only if
  // some pixel in it is rock, and the fine one is where that pixel would be.
  const coarse = vec.patches2.length ? vec.patches.subarray(0, 61 * 107) : null;
  assert.ok(coarse.some((byte) => (byte & 3) !== 2), "some terrain in view");
  assert.ok(vec.patches2.subarray(0, 121 * 213).some((byte) => (byte & 3) !== 2));
});
