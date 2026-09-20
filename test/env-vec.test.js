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
import { EPISODE_STATS, HEADS, VecWormEnv } from "../src/env/vec.js";
import { PATCH_CELLS } from "../src/env/observation.js";

const absent = Object.values(ASSETS).filter(
  (name) => !existsSync(new URL(name, DEFAULT_ENGINE_DIR)),
);
const skip = absent.length
  ? `no engine kit in ${DEFAULT_ENGINE_DIR.pathname}: missing ${absent.join(", ")} — run artifacts/headless-sim/fetch-assets.sh`
  : false;

test("the heads are the keys a person would press, factorised", () => {
  assert.equal(HEADS, 7);
  assert.deepEqual(ACTION_SIZES, [3, 3, 2, 2, 2, 3, 3]);
  // Left and right together is the engine doing nothing, so they share a head
  // rather than being two bits that can contradict each other.
  assert.deepEqual(actionFromHeads([1, 0, 0, 0, 0, 0, 0]), {
    keys: KEYS.left,
    rope: ROPE.none,
    weapon: 0,
  });
  assert.deepEqual(actionFromHeads([2, 2, 1, 1, 1, 2, 2]), {
    keys: KEYS.right | KEYS.aimDown | KEYS.fire | KEYS.jump | KEYS.dig,
    rope: ROPE.release,
    weapon: -1,
  });
  // Read out of the middle of a batch, which is how they arrive.
  const batch = Uint8Array.from([0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1]);
  assert.equal(actionFromHeads(batch, 7).keys, KEYS.right | KEYS.aimUp | KEYS.fire);
  assert.equal(actionFromHeads(batch, 7).rope, ROPE.throw);
  assert.equal(actionFromHeads(batch, 7).weapon, 1);
  assert.equal(
    ACTION_HEADS.reduce((all, [, choices]) => all * choices.length, 1),
    648,
    "648 combinations, described by 18 numbers",
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
  let restarts = 0;
  for (let step = 0; step < 60; step++) {
    vec.step(heads);
    restarts += vec.dones.reduce((sum, one) => sum + one, 0);
  }
  assert.equal(restarts, 4, "each of the four matches ended once");
  const stats = Object.fromEntries(
    EPISODE_STATS.map((field, index) => [field, vec.stats[index]]),
  );
  assert.equal(stats.steps, 50, "200 ticks at 4 ticks a step");
  assert.ok(Number.isFinite(stats.reward));
  assert.ok(stats.seed > 0);
  assert.equal(vec.envs[0].done, false, "and is already playing the next one");
});

test("one policy can drive any number of worms", { skip }, async () => {
  const engine = await loadEngine();
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
    assert.equal(vec.describe().vectorSize, 103);
    assert.equal(vec.vectors.length, 2 * agents * 103);
    vec.step(new Uint8Array(2 * agents * HEADS));
  }
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
    assert.equal(layout.actionBytes, 2 * 3 * HEADS);
    assert.deepEqual(layout.order, ["vectors", "patches", "rewards", "dones", "stats"]);
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
    const rewards = new Float32Array(
      stepped.buffer.slice(
        stepped.byteOffset + layout.bytes.vectors + layout.bytes.patches,
        stepped.byteOffset + layout.bytes.vectors + layout.bytes.patches + layout.bytes.rewards,
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
