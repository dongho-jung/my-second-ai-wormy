import test from "node:test";
import assert from "node:assert/strict";
import { snapshotV20 } from "../src/adapter-v20.js";
import { fixture, LEVEL } from "./fixture.js";
import {
  CURRICULUM_DEFAULTS,
  GoalCurriculum,
  GoalDeadlineCurriculum,
} from "../src/env/curriculum.js";
import { groundedGoal, groundedGoalNear } from "../src/env/env.js";
import { VECTOR_OFFSETS, encodeVector } from "../src/env/observation.js";
import { solidAt, terrainOf } from "../src/env/terrain.js";
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

test("a near goal is within the radius, a few strides off, and standable", () => {
  // A flat 400x200 field with a floor at y=150: index 0 background, 1 dirt.
  const width = 400;
  const height = 200;
  const data = new Uint8Array(width * height);
  for (let y = 150; y < height; y++) for (let x = 0; x < width; x++) data[y * width + x] = 1;
  const materialFlags = new Uint8Array(256).fill(8);
  materialFlags[1] = 3;
  const terrain = terrainOf({ data, width, height, materialFlags });
  const rng = rngOver(Array.from({ length: 997 }, (_, i) => ((i * 7919) % 997) / 997));
  const from = { x: 200, y: 140 };

  let found = 0;
  for (let attempt = 0; attempt < 40; attempt++) {
    const goal = groundedGoalNear(terrain, rng, from, 100);
    if (!goal) continue;
    found++;
    const distance = Math.hypot(goal.x - from.x, goal.y - from.y);
    assert.ok(distance <= 100, `goal ${distance.toFixed(0)}px away is past the radius`);
    assert.ok(distance >= 40, `goal ${distance.toFixed(0)}px away is underfoot`);
    assert.equal(solidAt(terrain, goal.x, goal.y), false, "goal stands in the floor");
    assert.ok(goal.y + 48 >= 150, "goal floats with nothing under it");
  }
  assert.ok(found >= 30, `a hundred-pixel radius on open ground should nearly always find a spot, found ${found}`);
  // Asked for somewhere above the worm, only rows at least that far up are
  // drawn from — here the floor is below, so the sky above it, standing on
  // nothing, is refused and there is nothing to offer.
  assert.equal(groundedGoalNear(terrain, rng, from, 100, { abovePx: 48 }), null);
  // From down on the floor with a ledge above, the ledge is what comes back.
  for (let y = 100; y < 110; y++) for (let x = 150; x < 250; x++) data[y * width + x] = 1;
  let above = 0;
  for (let attempt = 0; attempt < 40; attempt++) {
    const goal = groundedGoalNear(terrain, rng, { x: 200, y: 140 }, 100, { abovePx: 48 });
    if (!goal) continue;
    above++;
    assert.ok(140 - goal.y >= 48, `goal at y=${goal.y} is not 48px above the worm at 140`);
  }
  assert.ok(above >= 20, `a ledge within reach should be found most of the time, found ${above}`);
  // A radius shorter than the closest allowed goal has nothing to offer.
  assert.equal(groundedGoalNear(terrain, rng, from, 30), null);
  // And nowhere standable within reach is null rather than a guess: the sky.
  assert.equal(groundedGoalNear(terrain, rng, { x: 200, y: 10 }, 60), null);
});

test("the radius moves out when the worms keep arriving, and back when they do not", () => {
  const curriculum = new GoalCurriculum({ from: 96, to: 1600, window: 10 });
  assert.equal(curriculum.radius, 96, "starts at the near end");
  // Nine outcomes are not a window: nothing moves, however good they were.
  for (let i = 0; i < 9; i++) curriculum.record(true);
  assert.equal(curriculum.radius, 96);
  // The tenth completes it: nine of ten reached is above the 85% bar.
  curriculum.record(true);
  assert.ok(Math.abs(curriculum.radius - 96 * 1.1) < 1e-9, "one notch out");
  // Six of ten is between the bars: stays.
  for (let i = 0; i < 10; i++) curriculum.record(i < 6);
  assert.ok(Math.abs(curriculum.radius - 96 * 1.1) < 1e-9, "middling outcomes hold it");
  // Four of ten is at or below half: one notch back, and never below `from`.
  for (let i = 0; i < 10; i++) curriculum.record(i < 4);
  assert.ok(Math.abs(curriculum.radius - 96) < 1e-9, "one notch back");
  for (let i = 0; i < 10; i++) curriculum.record(false);
  assert.equal(curriculum.radius, 96, "it does not go under the near end");
  // Nor over the far end, however long the run of good windows.
  const far = new GoalCurriculum({ from: 96, to: 120, window: 2, start: 115 });
  assert.equal(far.radius, 115, "a carried-on run starts where it had got to");
  far.record(true);
  far.record(true);
  assert.equal(far.radius, 120, "capped at the far end");
  // A start outside the range is pulled inside it.
  assert.equal(new GoalCurriculum({ from: 96, to: 120, start: 5000 }).radius, 120);
  assert.equal(CURRICULUM_DEFAULTS.window, 30);
});

test("a reliable worm gets less time, and gets time back when it misses", () => {
  const curriculum = new GoalDeadlineCurriculum({ from: 450, to: 120, window: 10 });
  for (let i = 0; i < 10; i++) curriculum.record(true);
  assert.ok(Math.abs(curriculum.patience - 450 / 1.1) < 1e-9, "one notch tighter");
  for (let i = 0; i < 10; i++) curriculum.record(i < 6);
  assert.ok(Math.abs(curriculum.patience - 450 / 1.1) < 1e-9, "middling outcomes hold it");
  for (let i = 0; i < 10; i++) curriculum.record(false);
  assert.equal(curriculum.patience, 450, "failure loosens it, but never past the start");

  const near = new GoalDeadlineCurriculum({ from: 450, to: 120, start: 121, window: 2 });
  near.record(true);
  near.record(true);
  assert.equal(near.patience, 120, "never tighter than the requested floor");
  assert.throws(
    () => new GoalDeadlineCurriculum({ from: 100, to: 200 }),
    /from >= to > 0/,
  );
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
