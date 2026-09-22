import test from "node:test";
import assert from "node:assert/strict";
import { ROPE } from "../src/env/actions.js";
import { DEFAULTS } from "../src/env/env.js";

// The cooldown is the one thing here that changes what the engine is told, so
// what it lets through and what it swallows is worth pinning down. The counting
// itself is exercised by a run; these are the rules it counts under.

/** The gate in `step`, on its own: which decisions carry a rope message. */
function heard(cooldown, wanted) {
  let readyAt = 0;
  const out = [];
  wanted.forEach((rope, decision) => {
    if (rope === ROPE.none) {
      out.push(ROPE.none);
      return;
    }
    if (cooldown > 0 && decision < readyAt) {
      out.push(ROPE.none);
      return;
    }
    if (cooldown > 0) readyAt = decision + cooldown;
    out.push(rope);
  });
  return out;
}

test("no cooldown hears every rope message", () => {
  const wanted = [ROPE.throw, ROPE.release, ROPE.throw, ROPE.throw];
  assert.deepEqual(heard(0, wanted), wanted);
});

test("a cooldown hears one message and swallows the rest of the window", () => {
  const wanted = Array.from({ length: 12 }, () => ROPE.throw);
  const out = heard(5, wanted);
  assert.deepEqual(
    out.map((one) => (one === ROPE.throw ? 1 : 0)),
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    "a message should land on decisions 0, 5 and 10",
  );
});

test("letting go costs the same window as throwing", () => {
  // Otherwise a worm could throw, release immediately, and throw again — which
  // is the thrashing the cooldown exists to slow down.
  const out = heard(4, [ROPE.throw, ROPE.release, ROPE.release, ROPE.release, ROPE.release]);
  assert.deepEqual(out, [ROPE.throw, ROPE.none, ROPE.none, ROPE.none, ROPE.release]);
});

test("decisions that ask for nothing do not start a window", () => {
  const out = heard(3, [ROPE.none, ROPE.none, ROPE.throw, ROPE.throw]);
  assert.deepEqual(out, [ROPE.none, ROPE.none, ROPE.throw, ROPE.none]);
});

test("the default is off, so runs that say nothing behave as before", () => {
  assert.equal(DEFAULTS.ropeCooldown, 0);
});
