import test from "node:test";
import assert from "node:assert/strict";
import { snapshotV20 } from "../src/adapter-v20.js";
import { fixture, LEVEL } from "./fixture.js";
import {
  DEFAULT_SPEC,
  PATCH,
  PATCH_CELLS,
  PATCH_KIND,
  PATCH_PROJECTILE,
  PATCH_SIZE,
  VECTOR_OFFSETS,
  VECTOR_SIZE,
  encodePatch,
  encodePatchBytes,
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
  const span = (name, size) =>
    Array.from(vector.subarray(VECTOR_OFFSETS[name], VECTOR_OFFSETS[name] + size));
  assert.deepEqual(span("contacts", 4), [0, 0, f32(3 / 7), 0]);
  assert.equal(at("stepping"), 0);
  assert.deepEqual(span("walkLeft", 4), [1, 0, 0, 0], "clear to the left");
  assert.deepEqual(span("walkRight", 4), [0, 0, 0, 1], "rock to the right");

  // The one weapon is out of ammo and selected, so it is not ready to fire.
  assert.deepEqual(span("weapons", 15).slice(0, 3), [0, 0, 1]);
  assert.deepEqual(
    span("weapons", 15).slice(3),
    new Array(12).fill(0),
    "the four empty slots read as nothing, not as ready",
  );
  assert.deepEqual(span("rope", 5), [0, 0, 0, 0, 0], "no rope out");
  const FOE_WIDTH = DEFAULT_SPEC.layout.find(([name]) => name === "foes")[1] / DEFAULT_SPEC.foeSlots;
  assert.deepEqual(
    span("foes", DEFAULT_SPEC.foeSlots * FOE_WIDTH),
    new Array(DEFAULT_SPEC.foeSlots * FOE_WIDTH).fill(0),
    "a dead foe is all zeros, starting with the alive flag",
  );
  // The fixture's weapon is not one the profiler knows, so its character reads
  // as nothing rather than as something invented.
  assert.deepEqual(span("weapon", 10), new Array(10).fill(0));
  // 38 ticks left of a reload, against the 300 the field is scaled by.
  assert.equal(span("weaponsReload", 5)[0], Math.fround(38 / 300));
  assert.deepEqual(span("weaponsReload", 5).slice(1), [0, 0, 0, 0]);
  assert.deepEqual(
    span("pickups", 12).slice(0, 4),
    [Math.fround((2 - 32) / 300), Math.fround((3 - 26) / 300), 1, 0],
    "the health crate the fixture holds",
  );
  // One shot, ten pixels left and two below, and two empty slots after it.
  assert.equal(Math.round(at("projectiles", 0) * 300), -10);
  assert.equal(Math.round(at("projectiles", 1) * 300), 2);
  assert.equal(at("projectiles", 2), 2);
  assert.equal(at("projectiles", 3), -3);
  assert.deepEqual(
    span("projectiles", 24).slice(8),
    new Array(16).fill(0),
    "the two empty shot slots stay empty",
  );
});

test("both foes appear, nearest first, in a free-for-all", () => {
  const view = liveView();
  const worm = (x, y, health) => ({
    alive: true,
    position: { x, y },
    velocity: { x: 1, y: -2 },
    health,
  });
  // The far one is listed first, to prove the ordering is by distance.
  view.foes = [worm(32 + 400, 26, 90), worm(32 + 30, 26 - 40, 50)];
  const vector = encodeVector(view);
  const width = DEFAULT_SPEC.layout.find(([name]) => name === "foes")[1] / DEFAULT_SPEC.foeSlots;
  const slot = (index) =>
    vector.subarray(
      VECTOR_OFFSETS.foes + index * width,
      VECTOR_OFFSETS.foes + (index + 1) * width,
    );
  const near = slot(0);
  assert.equal(near[0], 1);
  assert.equal(Math.round(near[1] * 300), 30);
  assert.equal(Math.round(near[2] * 300), -40);
  assert.equal(Math.round(near[3] * 300), 50, "50 pixels away");
  assert.equal(+near[4].toFixed(4), 0.6, "unit direction, so clipping cannot hide it");
  assert.equal(+near[5].toFixed(4), -0.8);
  assert.equal(near[6], 0.5);
  assert.ok(near[3] < 1, "50 of 300 pixels is well inside the clip");
  assert.deepEqual([near[7], near[8]], [1, -2]);
  const far = slot(1);
  assert.equal(far[0], 1);
  assert.equal(far[6], Math.fround(0.9), "the second slot is the other worm");
  // 400 pixels is past the 300 the vector measures against, so the distance and
  // the offset both clip — and the unit direction underneath still points at it.
  assert.equal(far[1], 1);
  assert.equal(far[3], 1);
  assert.equal(far[4], 1, "still due right");

  // One foe left alive fills the first slot and leaves the second empty.
  view.foes[0].alive = false;
  const alone = encodeVector(view);
  assert.equal(alone[VECTOR_OFFSETS.foes], 1);
  assert.equal(alone[VECTOR_OFFSETS.foes + width], 0);
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

test("the byte patch and the one-hot patch are the same picture", () => {
  const view = liveView();
  const bytes = encodePatchBytes(view);
  const floats = encodePatch(view);
  assert.equal(bytes.length, PATCH_CELLS);
  assert.equal(floats.length, PATCH_SIZE);
  // A quarter of the bytes, because one cell is one byte and not four floats.
  assert.equal(bytes.byteLength * 4, floats.byteLength / 4);
  for (let cell = 0; cell < PATCH_CELLS; cell++) {
    const kind = bytes[cell] & PATCH_KIND;
    assert.ok(kind <= 2, `cell ${cell} has a terrain code of ${kind}`);
    for (let channel = 0; channel < 3; channel++) {
      assert.equal(
        floats[channel * PATCH_CELLS + cell],
        channel === kind ? 1 : 0,
        `cell ${cell} channel ${channel}`,
      );
    }
    assert.equal(
      floats[3 * PATCH_CELLS + cell],
      bytes[cell] & PATCH_PROJECTILE ? 1 : 0,
    );
  }
  const shots = bytes.reduce((sum, byte) => sum + (byte & PATCH_PROJECTILE ? 1 : 0), 0);
  assert.equal(shots, 1, "the one live shot");
  // A dead worm writes zeros, and zero is rock — which is why the alive flag in
  // the vector is what says the patch means anything.
  const dead = encodePatchBytes({ ...view, self: { alive: false } });
  assert.deepEqual(Array.from(dead), new Array(PATCH_CELLS).fill(0));
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
  assert.equal(into.vector[VECTOR_OFFSETS.foes], 1);
  view.foes[0] = { alive: false };
  observe(view, into);
  assert.equal(into.vector[VECTOR_OFFSETS.foes], 0, "the stale foe is gone");
});
