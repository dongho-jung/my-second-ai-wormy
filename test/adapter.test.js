import test from "node:test";
import assert from "node:assert/strict";
import { snapshotV20 } from "../src/adapter-v20.js";
import { fixture, LEVEL } from "./fixture.js";

test("state keeps coordinates, scores, ammo and null spectator worms, detached", () => {
  const controller = fixture();
  const before = JSON.stringify(controller.Ub.X.F);
  const state = snapshotV20.call(controller);
  assert.equal(state.tick, 120);
  assert.equal(state.localPlayerId, 12);
  assert.equal(state.room.mode, "deathmatch");
  assert.equal(state.match.elapsedSeconds, 2);
  assert.equal(state.match.remainingSeconds, 598);
  assert.deepEqual(state.map, {
    name: "Fixture",
    width: LEVEL.width,
    height: LEVEL.height,
  });
  const self = state.players[0];
  assert.deepEqual(self.worm.position, { x: 32, y: 26 });
  assert.deepEqual(self.worm.velocity, { x: -1.25, y: 0.5 });
  assert.equal(self.worm.health, 67);
  assert.equal(self.worm.keys, 17, "the engine's own input bitmask");
  assert.equal(self.worm.aimRadians, -0.25);
  assert.equal(self.worm.aimVelocity, -0.02, "turning in the same screen angle");
  assert.equal(self.worm.weapons[0].reloadTicksRemaining, 38);
  assert.deepEqual(self.score, { value: 3, display: "3", kills: 5, deaths: 2 });
  assert.equal(state.players[1].worm, null);
  assert.equal(state.players[1].alive, false);
  assert.equal(
    state.projectiles.length,
    1,
    "inactive and out-of-range pool slots must be excluded",
  );
  assert.equal(state.pickups[0].kind, "health");
  assert.equal(JSON.stringify(controller.Ub.X.F), before, "reads never mutate");
  controller.Ub.X.B.get(12).ra.x = 90;
  assert.equal(state.players[0].worm.position.x, 32, "snapshots are detached");
});

test("death, room exit and left-facing aim", () => {
  const controller = fixture();
  controller.Ub.X.B.get(12).ra.direction = 0;
  let state = snapshotV20.call(controller);
  assert.equal(state.players[0].worm.aimRadians, Math.PI + 0.25);
  assert.equal(state.players[0].worm.aimVelocity, 0.02);
  controller.Ub.X.B.get(12).ra.u = false;
  state = snapshotV20.call(controller);
  assert.equal(state.players[0].worm, null);
  controller.w.m.isConnected = false;
  assert.equal(snapshotV20.call(controller), null);
});

test("the near patch is one byte per pixel with the worm on the centre one", () => {
  const { patch, tick, localPlayerId } = snapshotV20.call(fixture(), {
    terrainPatch: true,
  });
  assert.equal(tick, 120);
  assert.equal(localPlayerId, 12);
  assert.deepEqual(patch.self, { x: 32, y: 26 });
  const { near } = patch;
  const [width, height] = near.size;
  assert.deepEqual(near.size, [427, 241], "odd sides give the worm one pixel");
  assert.deepEqual(near.bounds, [LEVEL.width, LEVEL.height]);
  assert.deepEqual(
    near.origin,
    [32 - (width >> 1), 26 - (height >> 1)],
    "the window is anchored on the worm",
  );
  const bytes = Buffer.from(near.data, "base64");
  assert.equal(bytes.length, width * height);
  // Every byte is the level's own palette index, read straight through.
  const at = (x, y) => bytes[(y - near.origin[1]) * width + (x - near.origin[0])];
  assert.equal(at(32, 26), 0, "the worm stands in open air");
  assert.equal(at(32, LEVEL.floorY), 1, "the floor is dirt");
  assert.equal(at(LEVEL.wallX, 26), 2, "the wall is rock");
  // Past the edge of the level there is nothing to read; `bounds` is what says
  // so, rather than a sentinel index a level could legitimately use.
  assert.equal(at(-5, -5), 0);
});

test("the patch measures the ground, the walls and the engine's own contacts", () => {
  const { patch } = snapshotV20.call(fixture(), { terrainPatch: true });
  // The worm stands four pixels above the floor with nothing overhead.
  assert.equal(patch.overhead.groundPx, LEVEL.floorY - 26);
  assert.equal(patch.overhead.ceilingPx, null);
  assert.equal(patch.overhead.rightPx, LEVEL.wallX - 32);
  assert.equal(patch.overhead.leftPx, 32 - 5);
  assert.deepEqual(patch.contacts, {
    up: 0,
    down: 3,
    left: 0,
    right: 0,
    stepping: false,
  });
  // Rock ahead is a wall for good; the open side is a plain step.
  assert.equal(patch.walk.right, "rock");
  assert.equal(patch.walk.left, "clear");
});

test("a dead or spectating local player has no surroundings", () => {
  const controller = fixture();
  controller.Ub.X.B.get(12).ra.u = false;
  const read = snapshotV20.call(controller, { terrainPatch: true });
  assert.equal(read.patch, null);
  assert.equal(read.tick, 120);
});

test("terrain preserves all row-major bytes and both lookup tables", () => {
  const state = snapshotV20.call(fixture(), { terrain: true });
  const bytes = Buffer.from(state.data, "base64");
  assert.equal(bytes.length, LEVEL.width * LEVEL.height);
  assert.equal(bytes[0], 2, "the left edge is rock");
  assert.equal(bytes[LEVEL.floorY * LEVEL.width + 32], 1, "the floor is dirt");
  assert.equal(state.materialFlags.length, 256);
  assert.equal(state.paletteRgb.length, 768);
  assert.equal(state.order, "row-major");
});

test("invalid field mappings and malformed buffers fail visibly", () => {
  const controller = fixture();
  controller.Ub.X.B.get(12).ra.Xa = undefined;
  assert.throws(() => snapshotV20.call(controller), /missing game field/);
  controller.Ub.X.F.level.width = 7;
  assert.throws(
    () => snapshotV20.call(controller, { terrain: true }),
    /terrain buffer/,
  );
  assert.throws(
    () => snapshotV20.call({ w: { m: { isConnected: true } }, Ub: { X: {} } }),
    /no longer matches the adapter/,
  );
});
