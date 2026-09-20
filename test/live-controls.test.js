import test from "node:test";
import assert from "node:assert/strict";
import { KEYS, ROPE } from "../src/env/actions.js";
import { Controls, ControlsError } from "../src/live/controls.js";
import { BINDINGS } from "../src/live/keys.js";

/** A page that records what the keyboard was told, without a browser. */
function fakePage(stored = BINDINGS) {
  const events = [];
  return {
    events,
    keyboard: {
      down: async (code) => void events.push(`down ${code}`),
      up: async (code) => void events.push(`up ${code}`),
    },
    evaluate: async (fn, fallback) => (stored === null ? fallback : stored),
  };
}

test("a decision is the difference from the last one, not a fresh press", async () => {
  const page = fakePage();
  const controls = new Controls(page);
  await controls.verify();
  await controls.apply({ keys: KEYS.right | KEYS.fire });
  assert.deepEqual(page.events.sort(), ["down ArrowRight", "down KeyD"].sort());
  page.events.length = 0;

  // Still walking right, still firing: nothing should be pressed again. Jump
  // and dig only fire on the press, so re-pressing every decision would jump
  // fifteen times a second instead of once.
  await controls.apply({ keys: KEYS.right | KEYS.fire });
  assert.deepEqual(page.events, []);

  await controls.apply({ keys: KEYS.left });
  assert.deepEqual(page.events, ["up ArrowRight", "up KeyD", "down ArrowLeft"]);
});

test("holding jump is one jump, the same as in training", async () => {
  const page = fakePage();
  const controls = new Controls(page);
  await controls.verify();
  await controls.apply({ keys: KEYS.jump });
  await controls.apply({ keys: KEYS.jump });
  await controls.apply({ keys: KEYS.jump });
  assert.deepEqual(page.events, ["down KeyS"], "pressed once and left held");
});

test("the rope and the weapon are taps, and they come back up", async () => {
  const page = fakePage();
  const controls = new Controls(page, { tapMs: 5 });
  await controls.verify();
  await controls.apply({ keys: 0, rope: ROPE.throw, weapon: 1 });
  assert.deepEqual(page.events, ["down KeyA", "down KeyQ"]);
  await new Promise((resolve) => setTimeout(resolve, 25));
  assert.deepEqual(page.events.slice(2).sort(), ["up KeyA", "up KeyQ"].sort());
});

test("releasing lets go of everything it was holding", async () => {
  const page = fakePage();
  const controls = new Controls(page);
  await controls.verify();
  await controls.apply({ keys: KEYS.right | KEYS.aimUp | KEYS.dig });
  page.events.length = 0;
  await controls.release();
  assert.deepEqual(page.events.sort(), ["up ArrowRight", "up ArrowUp", "up KeyC"].sort());
  assert.equal(controls.held.size, 0);
});

test("a game whose settings disagree is refused, not driven blindly", async () => {
  const broken = { ...BINDINGS };
  delete broken.KeyD;
  const controls = new Controls(fakePage(broken));
  await assert.rejects(() => controls.verify(), ControlsError);
  await assert.rejects(() => controls.verify(), /Fire/);
  // And the real settings are read from the game rather than assumed.
  const moved = { ...BINDINGS, KeyF: "Fire" };
  delete moved.KeyD;
  const other = new Controls(fakePage(moved));
  assert.equal((await other.verify()).Fire, "KeyF");
});
