import test from "node:test";
import assert from "node:assert/strict";
import { snapshotV20 } from "../src/adapter-v20.js";
import { fixture, LEVEL } from "./fixture.js";
import { groundedGoal } from "../src/env/env.js";
import { VECTOR_OFFSETS, encodeVector } from "../src/env/observation.js";
import { solidAt } from "../src/env/terrain.js";
import { viewFromSnapshot } from "../src/env/view.js";

// Paying a worm for closing on a point it cannot see is paying at random, so
// what the vector says about the goal is the part worth pinning down.

function liveView() {
  const controller = fixture();
  return viewFromSnapshot(
    snapshotV20.call(controller),
    snapshotV20.call(controller, { terrain: true }),
  );
}

/** A generator that walks a fixed list, so a failure is the same every time. */
function rngOver(values) {
  let at = 0;
  return () => values[at++ % values.length];
}

test("a goal is background with ground under it, never inside the rock", () => {
  const { terrain } = liveView();
  // Enough draws to cover the level, including the rock column on the left and
  // the wall, which must never be chosen.
  const draws = [];
  for (let i = 0; i < 400; i++) draws.push((i * 37) % 100 / 100);
  const rng = rngOver(draws);

  let found = 0;
  for (let attempt = 0; attempt < 50; attempt++) {
    const goal = groundedGoal(terrain, rng);
    if (!goal) continue;
    found++;
    assert.equal(solidAt(terrain, goal.x, goal.y), false, "goal stands in solid ground");
    let ground = false;
    for (let below = 1; below <= 48 && !ground; below++) {
      if (solidAt(terrain, goal.x, goal.y + below)) ground = true;
    }
    assert.equal(ground, true, "goal floats with nothing under it");
    assert.notEqual(goal.x, LEVEL.wallX, "goal sits in the wall");
  }
  assert.ok(found > 0, "no goal was found on a level that has a floor");
});

test("no goal leaves the four values at zero", () => {
  const view = liveView();
  view.goal = null;
  const vector = encodeVector(view);
  const at = VECTOR_OFFSETS.goal;
  assert.deepEqual([...vector.slice(at, at + 4)], [0, 0, 0, 0]);
});

test("a goal reads as a unit direction and a saturating distance", () => {
  const view = liveView();
  const { x, y } = view.self.position;
  // Straight down and to the right, 300px away: 180-240-300.
  view.goal = { x: x + 180, y: y + 240 };
  const vector = encodeVector(view);
  const at = VECTOR_OFFSETS.goal;

  assert.equal(vector[at], 1, "the set flag is not raised");
  assert.ok(Math.abs(vector[at + 1] - 0.6) < 1e-6, "x direction is not a unit vector");
  assert.ok(Math.abs(vector[at + 2] - 0.8) < 1e-6, "y direction is not a unit vector");
  assert.ok(Math.abs(vector[at + 3] - 300 / 800) < 1e-6, "distance is not scaled");

  // Past the scale the direction still has to be exact; only the distance pins.
  view.goal = { x: x + 4000, y };
  const far = encodeVector(view);
  assert.equal(far[at + 1], 1, "x direction stopped being a unit vector when far");
  assert.equal(far[at + 3], 1, "distance did not saturate");
});

test("a dead worm reports no goal, like it reports nothing else", () => {
  const view = liveView();
  view.goal = { x: 10, y: 10 };
  view.self = { ...view.self, alive: false };
  const vector = encodeVector(view);
  const at = VECTOR_OFFSETS.goal;
  assert.deepEqual([...vector.slice(at, at + 4)], [0, 0, 0, 0]);
});
