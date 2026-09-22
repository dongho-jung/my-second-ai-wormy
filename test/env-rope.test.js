import test from "node:test";
import assert from "node:assert/strict";
import { KEYS, ROPE, RopeHold } from "../src/env/actions.js";
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
  assert.equal(DEFAULTS.ropeHold, 0);
});

test("a throw is kept: no re-throw and no jump until the hold is over", () => {
  const hold = new RopeHold(3);
  const throwing = { keys: KEYS.right, rope: ROPE.throw, weapon: 0 };
  const jumping = { keys: KEYS.right | KEYS.jump, rope: ROPE.none, weapon: 0 };
  assert.equal(hold.share, 0);
  assert.deepEqual(hold.apply(throwing), throwing, "the throw itself goes through");
  assert.equal(hold.share, 1, "and the hold starts");
  // For three decisions the rope is left alone: a throw is swallowed, and the
  // jump key — which is how a rope is let go — is dropped. Walking is kept.
  assert.deepEqual(hold.apply(throwing), { keys: KEYS.right, rope: ROPE.none, weapon: 0 });
  assert.deepEqual(hold.apply(jumping), { keys: KEYS.right, rope: ROPE.none, weapon: 0 });
  assert.ok(Math.abs(hold.share - 1 / 3) < 1e-9);
  assert.deepEqual(hold.apply(jumping), { keys: KEYS.right, rope: ROPE.none, weapon: 0 });
  assert.equal(hold.share, 0);
  // The fourth decision is free again: this jump lets go.
  assert.deepEqual(hold.apply(jumping), jumping);
  // And a fresh throw starts a fresh hold.
  assert.deepEqual(hold.apply(throwing), throwing);
  assert.equal(hold.share, 1);
  hold.reset();
  assert.equal(hold.share, 0);
  // No hold configured: everything goes through untouched.
  const none = new RopeHold(0);
  assert.deepEqual(none.apply(throwing), throwing);
  assert.deepEqual(none.apply(jumping), jumping);
});
