import test from "node:test";
import assert from "node:assert/strict";
import { ACTION_HEADS, ACTION_SIZES, KEYS, actionFromHeads } from "../src/env/actions.js";
import {
  ACTIONS,
  bitsFromKeys,
  headsFromKeys,
  keysForBits,
  pressesForAction,
} from "../src/live/keys.js";

/** Every action the policy's eight heads can produce. */
function* everyAction() {
  const counters = new Array(ACTION_SIZES.length).fill(0);
  for (;;) {
    yield actionFromHeads(counters);
    let head = counters.length - 1;
    while (head >= 0) {
      counters[head]++;
      if (counters[head] < ACTION_SIZES[head]) break;
      counters[head--] = 0;
    }
    if (head < 0) return;
  }
}

test("every action the policy can take is one a player could press", () => {
  let checked = 0;
  for (const action of everyAction()) {
    const held = keysForBits(action.keys);
    assert.equal(
      bitsFromKeys(held),
      action.keys,
      `holding ${[...held].join("+") || "nothing"} gives ${bitsFromKeys(held)}, wanted ${action.keys}`,
    );
    checked++;
  }
  assert.equal(checked, ACTION_SIZES.reduce((all, size) => all * size, 1));
  assert.equal(checked, 1944);
});

test("the two ways of reaching dig do not collide", () => {
  // A player with no dig key bound digs by holding left and right at once, so
  // the engine reads that pair as 256. An action space that let a policy ask
  // for both would be asking for something else entirely.
  assert.equal(bitsFromKeys(new Set([ACTIONS.left, ACTIONS.right])), 1 | 2 | 256);
  assert.throws(() => keysForBits(KEYS.left | KEYS.right), /dig/);
  // Which is why move is one three-way choice and not two bits.
  for (const action of everyAction()) {
    assert.notEqual(
      action.keys & (KEYS.left | KEYS.right),
      KEYS.left | KEYS.right,
      "the heads must never ask for both",
    );
  }
  assert.deepEqual(keysForBits(KEYS.dig), new Set([ACTIONS.dig]));
});

test("ChangeWeap is a modifier, and nothing we press needs it", () => {
  // Held down, it turns aim into rope length and stops the worm walking.
  assert.equal(bitsFromKeys(new Set([ACTIONS.changeWeapon, ACTIONS.up])), 4 | 64);
  assert.equal(bitsFromKeys(new Set([ACTIONS.changeWeapon, ACTIONS.down])), 8 | 128);
  assert.equal(bitsFromKeys(new Set([ACTIONS.changeWeapon, ACTIONS.left, ACTIONS.jump])), 0);
  // So we bind a key per bit instead and never hold it.
  for (const action of everyAction()) {
    assert.ok(
      !keysForBits(action.keys).has(ACTIONS.changeWeapon),
      "no action should need the modifier",
    );
  }
});

test("the rope and the weapon are presses, not holds", () => {
  assert.deepEqual(pressesForAction({ rope: 1 }), [ACTIONS.rope]);
  // Letting go is the Jump key: the client only ever sends a throw on the
  // NinjaRope key, and the release on a press of Jump.
  assert.deepEqual(pressesForAction({ rope: -1 }), [ACTIONS.jump]);
  assert.deepEqual(pressesForAction({ weapon: 1 }), [ACTIONS.nextWeapon]);
  assert.deepEqual(pressesForAction({ weapon: -1 }), [ACTIONS.previousWeapon]);
  assert.deepEqual(pressesForAction({}), []);
});

test("the bindings cover every action a policy can ask for", async () => {
  const { BINDINGS, REQUIRED_ACTIONS, codesByAction, missingBindings } = await import(
    "../src/live/keys.js"
  );
  assert.deepEqual(missingBindings(BINDINGS), [], "every action needs a key");
  const codes = codesByAction(BINDINGS);
  assert.equal(codes.Fire, "KeyD");
  assert.equal(codes.Jump, "KeyS");
  assert.equal(codes.Dig, "KeyC");
  assert.equal(codes.NinjaRope, "KeyA");
  assert.equal(codes.Left, "ArrowLeft");
  assert.equal(codes.ShortenRope, "ShiftLeft");
  // Nothing is on ChangeWeap, so the modifier cannot be pressed by accident.
  assert.equal(codes.ChangeWeap, undefined);
  assert.ok(!Object.values(BINDINGS).includes("ChangeWeap"));
  // Nor on rope length: the seven heads have no way to ask for it, so a key
  // that could only ever be pressed by accident is better left unbound.
  assert.equal(codes.LengthenRope, undefined);
  // Every key a policy could need is one of the bound ones.
  for (const action of REQUIRED_ACTIONS) assert.ok(codes[action], `${action} is unbound`);
  // A table that has lost a key says which.
  const broken = { ...BINDINGS };
  delete broken.KeyC;
  assert.deepEqual(missingBindings(broken), ["Dig"]);
});

test("a person's keys read back as every one of the policy's heads", () => {
  // The recording is the policy's own action space, or it teaches nothing:
  // one head short and the weapon choice lands in the rope-length column.
  for (const action of everyAction()) {
    const heads = headsFromKeys(action.keys, action);
    assert.equal(heads.length, ACTION_HEADS.length);
    assert.deepEqual(actionFromHeads(heads), action);
  }
  assert.equal(headsFromKeys(KEYS.ropeShorter)[6], 1, "shorten is the seventh head");
  assert.equal(headsFromKeys(KEYS.ropeLonger)[6], 2);
  assert.equal(headsFromKeys(KEYS.left | KEYS.right)[4], 1, "both is dig, not a contradiction");
});
