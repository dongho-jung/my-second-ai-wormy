import test from "node:test";
import assert from "node:assert/strict";
import { snapshotV20 } from "../src/adapter-v20.js";
import { fixture, LEVEL } from "./fixture.js";
import {
  PATCH,
  PATCH_SIZE,
  VECTOR_OFFSETS,
  VECTOR_SIZE,
  encodePatch,
  encodeVector,
  nearestProjectiles,
  observe,
  patchCellOf,
  patchOriginOf,
} from "../src/env/observation.js";
import { viewFromSnapshot } from "../src/env/view.js";
import { contactsAt, reach, walkProbe } from "../src/env/terrain.js";

// The live game is the other source an observation is built from, so the whole
// encoder is exercised through it: if a mapping the adapter reports ever stops
// lining up with what the encoder wants, these fail rather than a first real
// match going quietly wrong.
function liveView() {
  const controller = fixture();
  return viewFromSnapshot(
    snapshotV20.call(controller),
    snapshotV20.call(controller, { terrain: true }),
  );
}

test("a live snapshot and a map read make one view", () => {
  const view = liveView();
  assert.deepEqual(view.self.position, { x: 32, y: 26 });
  assert.equal(view.self.health, 67);
  assert.equal(view.self.facing, "right");
  assert.equal(view.self.aimRadians, -0.25);
  assert.equal(view.self.weapons[0].reloadTicksRemaining, 38);
  assert.deepEqual(view.map, {
    name: "Fixture",
    width: LEVEL.width,
    height: LEVEL.height,
  });
  assert.equal(view.terrain.data.length, LEVEL.width * LEVEL.height);
  assert.equal(view.terrain.materialFlags.length, 256);
  assert.equal(view.foes.length, 1);
  assert.equal(view.foes[0].alive, false, "the other player is a spectator");
  assert.equal(view.projectiles.length, 1, "one live shot, not the dead slot");
});

test("the vector puts every field where the layout says", () => {
  const view = liveView();
  const vector = encodeVector(view);
  assert.equal(vector.length, VECTOR_SIZE);
  const at = (name, index = 0) => vector[VECTOR_OFFSETS[name] + index];
  // The vector is float32: an expected value has to be rounded to it first.
  const f32 = Math.fround;
  const rayPx = (index) => Math.round(at("rays", index) * 120);

  // Rays run clockwise from straight right, on a level with a rock wall at
  // x=39, rock from x=0 to 5, and a dirt floor at y=30.
  assert.equal(rayPx(0), LEVEL.wallX - 32, "right: the rock wall");
  assert.equal(rayPx(4), LEVEL.floorY - 26, "down: the floor");
  assert.equal(rayPx(8), 32 - 5, "left: the rock block");
  assert.equal(rayPx(12), 27, "up: off the top of the level");

  assert.equal(at("health"), f32(0.67));
  assert.equal(at("velocity", 0), -1.25);
  assert.equal(at("velocity", 1), 0.5);
  assert.equal(at("aim", 0), f32(Math.cos(-0.25)));
  assert.equal(at("aim", 1), f32(Math.sin(-0.25)));
  assert.equal(at("facing"), 1, "+1 right");

  // The engine's own contact counts: three probes on the ground, none else.
  assert.deepEqual(Array.from(vector.subarray(22, 26)), [0, 0, f32(3 / 7), 0]);
  assert.equal(at("stepping"), 0);
  assert.deepEqual(
    Array.from(vector.subarray(VECTOR_OFFSETS.walkLeft, VECTOR_OFFSETS.walkLeft + 4)),
    [1, 0, 0, 0],
    "clear to the left",
  );
  assert.deepEqual(
    Array.from(vector.subarray(VECTOR_OFFSETS.walkRight, VECTOR_OFFSETS.walkRight + 4)),
    [0, 0, 0, 1],
    "rock to the right",
  );

  // The one weapon is out of ammo and selected, so it is not ready to fire.
  assert.deepEqual(Array.from(vector.subarray(35, 38)), [0, 0, 1]);
  assert.deepEqual(
    Array.from(vector.subarray(38, 50)),
    new Array(12).fill(0),
    "the four empty slots read as nothing, not as ready",
  );
  assert.deepEqual(
    Array.from(vector.subarray(VECTOR_OFFSETS.rope, VECTOR_OFFSETS.rope + 5)),
    [0, 0, 0, 0, 0],
    "no rope out",
  );
  assert.deepEqual(
    Array.from(vector.subarray(VECTOR_OFFSETS.foe, VECTOR_OFFSETS.foe + 9)),
    new Array(9).fill(0),
    "a dead foe is all zeros, starting with the alive flag",
  );
  // One shot, ten pixels left and two below, and two empty slots after it.
  assert.equal(Math.round(at("projectiles", 0) * 300), -10);
  assert.equal(Math.round(at("projectiles", 1) * 300), 2);
  assert.equal(at("projectiles", 2), 2);
  assert.equal(at("projectiles", 3), -3);
  assert.deepEqual(Array.from(vector.subarray(68, 76)), new Array(8).fill(0));
});

test("the foe and its distance appear once it is alive", () => {
  const view = liveView();
  view.foes[0] = {
    alive: true,
    position: { x: 32 + 30, y: 26 - 40 },
    velocity: { x: 1, y: -2 },
    health: 50,
  };
  const vector = encodeVector(view);
  const foe = vector.subarray(VECTOR_OFFSETS.foe, VECTOR_OFFSETS.foe + 9);
  assert.equal(foe[0], 1);
  assert.equal(Math.round(foe[1] * 300), 30);
  assert.equal(Math.round(foe[2] * 300), -40);
  assert.equal(Math.round(foe[3] * 300), 50, "50 pixels away");
  assert.equal(+foe[4].toFixed(4), 0.6, "unit direction, so clipping cannot hide it");
  assert.equal(+foe[5].toFixed(4), -0.8);
  assert.equal(foe[6], 0.5);
  assert.equal(foe[3] < 1, true, "50 of 300 pixels is well inside the clip");
  assert.deepEqual([foe[7], foe[8]], [1, -2]);
});

test("a dead worm sees nothing at all", () => {
  const view = liveView();
  view.self = { alive: false };
  const { vector, patch } = observe(view);
  assert.deepEqual(Array.from(vector), new Array(VECTOR_SIZE).fill(0));
  assert.deepEqual(Array.from(patch), new Array(PATCH_SIZE).fill(0));
});

test("the patch is one hot channel per cell, with shots on their own", () => {
  const view = liveView();
  const patch = encodePatch(view);
  assert.equal(patch.length, PATCH_SIZE);
  const plane = PATCH.cells * PATCH.cells;
  const channel = (index) => patch.subarray(index * plane, (index + 1) * plane);
  const [rock, dirt, free, shots] = [0, 1, 2, 3].map(channel);
  for (let cell = 0; cell < plane; cell++) {
    assert.equal(
      rock[cell] + dirt[cell] + free[cell],
      1,
      `cell ${cell} must be exactly one of rock, dirt or free`,
    );
  }
  const origin = patchOriginOf(view);
  const cellAt = (x, y) => patchCellOf(origin, x, y).cell;
  // The worm is on the centre cell, in open air, with the dirt floor below.
  assert.equal(free[cellAt(32, 26)], 1);
  assert.equal(dirt[cellAt(32, LEVEL.floorY)], 1, "dirt underfoot");
  // The wall is one pixel wide and the patch samples every second pixel, so a
  // cell has to answer for both of them or the wall disappears.
  assert.equal(rock[cellAt(LEVEL.wallX, 26)], 1, "the rock wall");
  // Off the top of the level reads as rock, the way a wall does.
  assert.equal(rock[cellAt(32, -6)], 1);
  assert.equal(shots[cellAt(22, 28)], 1, "the one live shot");
  assert.equal(
    shots.reduce((sum, value) => sum + value, 0),
    1,
  );
});

test("only the nearest few shots reach the vector, nearest first", () => {
  const view = liveView();
  const shot = (x, y) => ({ position: { x, y }, velocity: { x: 0, y: 0 } });
  view.projectiles = [shot(32, 76), shot(32, 36), shot(32, 56), shot(32, 46)];
  assert.deepEqual(
    nearestProjectiles(view).map((one) => one.position.y),
    [36, 46, 56],
  );
});

test("the terrain helpers still agree with the adapter's own copy", () => {
  // The adapter inlines this arithmetic because it is stringified into the page,
  // so the repository holds two copies on purpose. They have to answer the same,
  // or the numbers an agent trains on stop being the numbers it will be given.
  const { patch } = snapshotV20.call(fixture(), { terrainPatch: true });
  const view = liveView();
  const { terrain } = view;
  const { x, y } = view.self.position;
  assert.deepEqual(contactsAt(terrain, x, y), patch.contacts);
  assert.equal(walkProbe(terrain, x, y, -1), patch.walk.left);
  assert.equal(walkProbe(terrain, x, y, 1), patch.walk.right);
  const limit = patch.overhead.ropeRangePx;
  assert.equal(reach(terrain, x, y, 0, -1, limit), patch.overhead.ceilingPx);
  assert.equal(reach(terrain, x, y, 0, 1, limit), patch.overhead.groundPx);
  assert.equal(reach(terrain, x, y, -1, 0, limit), patch.overhead.leftPx);
  assert.equal(reach(terrain, x, y, 1, 0, limit), patch.overhead.rightPx);
});

test("encoding into a reused buffer clears what was there before", () => {
  const view = liveView();
  const into = { vector: new Float32Array(VECTOR_SIZE), patch: new Float32Array(PATCH_SIZE) };
  const first = observe(view, into);
  assert.equal(first.vector, into.vector, "the buffer is written in place");
  view.foes[0] = {
    alive: true,
    position: { x: 40, y: 26 },
    velocity: { x: 0, y: 0 },
    health: 10,
  };
  observe(view, into);
  assert.equal(into.vector[VECTOR_OFFSETS.foe], 1);
  view.foes[0] = { alive: false };
  observe(view, into);
  assert.equal(into.vector[VECTOR_OFFSETS.foe], 0, "the stale foe is gone");
});
