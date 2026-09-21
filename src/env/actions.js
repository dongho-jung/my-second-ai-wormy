// What a player may do in a tick, in the engine's own terms.
//
// A room sends three different things, not one: a bitmask that is the state of
// the held keys, a rope message, and a weapon message. `room.ea` writes the
// bitmask straight onto `worm.Wa` and calls `world.update()`, and the two
// messages call worm methods of their own. Splitting them the same way here is
// what makes a policy trained against this environment mean the same thing when
// its output is turned back into key presses in a live game.
//
// The bit values were read off `worm.update()` in the v20 bundle and each one
// is covered by a test that watches the world react.

/** The held-key bitmask the engine reads once per tick as `worm.Wa`. */
export const KEYS = {
  left: 1,
  right: 2,
  aimUp: 4,
  aimDown: 8,
  fire: 16,
  jump: 32,
  ropeShorter: 64,
  ropeLonger: 128,
  // Carves away the dirt just in front of the worm. It is not weapon change:
  // that is a message of its own, below.
  dig: 256,
};

export const ALL_KEYS = Object.values(KEYS).reduce((all, bit) => all | bit, 0);

// Jump and dig fire on the press, not while held: the engine remembers whether
// the key was down last tick and does nothing until it has seen it released.
// A policy that holds either bit down jumps or digs exactly once.
export const EDGE_TRIGGERED = KEYS.jump | KEYS.dig;

/** Throw the rope, let it go, or leave it as it is. */
export const ROPE = { none: 0, throw: 1, release: -1 };

const KEY_NAMES = Object.keys(KEYS);

/** `{ right: true, fire: true }` -> `18`. */
export function packKeys(pressed = {}) {
  let bits = 0;
  for (const [name, down] of Object.entries(pressed)) {
    const bit = KEYS[name];
    if (bit === undefined) {
      throw new Error(`unknown key ${name}: expected ${KEY_NAMES.join(", ")}`);
    }
    if (down) bits |= bit;
  }
  return bits;
}

/** `18` -> `{ left: false, right: true, ... }`, every key named. */
export function unpackKeys(bits) {
  return Object.fromEntries(
    KEY_NAMES.map((name) => [name, (bits & KEYS[name]) !== 0]),
  );
}

/** `18` -> `"right+fire"`, for logs and test failures. */
export function describeKeys(bits) {
  const held = KEY_NAMES.filter((name) => bits & KEYS[name]);
  return held.length ? held.join("+") : "none";
}

/**
 * An action in its full form. A policy may emit just a bitmask, or a number,
 * and the rope and weapon messages default to sending nothing.
 */
export function normalizeAction(action = 0) {
  if (typeof action === "number") {
    return { keys: action & ALL_KEYS, rope: ROPE.none, weapon: 0 };
  }
  const keys = typeof action.keys === "number"
    ? action.keys & ALL_KEYS
    : packKeys(action.keys ?? {});
  const rope = action.rope ?? ROPE.none;
  if (rope !== ROPE.none && rope !== ROPE.throw && rope !== ROPE.release) {
    throw new Error(`rope must be one of ${Object.values(ROPE).join(", ")}`);
  }
  // The game's weapon message is a relative move through the five slots, so
  // that is what a policy learns to emit.
  const weapon = Math.trunc(action.weapon ?? 0);
  return { keys, rope, weapon };
}

/**
 * What a policy actually emits: eight small choices, not one number out of 1,944.
 *
 * Left and right at once is the engine doing nothing, so the two of them are one
 * three-way choice rather than two bits — and the same for aiming, and for the
 * rope's two length keys. The rest are the keys and messages that are genuinely
 * independent. Eight heads is twenty-one logits, against the 1,944 a single flat
 * distribution would need, and a policy that has learned to walk right keeps
 * that when it learns to fire.
 *
 * `ropeLength` is the one that was missing. Throwing the rope is a message and
 * was already here; reeling it in and paying it out are held keys — bits 64 and
 * 128, which `src/live/keys.js` already turns back into the game's own
 * shortenRope / lengthenRope bindings. Without this head a worm can fire the
 * rope and hang off it, and cannot swing, which is most of what the rope is for.
 */
export const ACTION_HEADS = [
  ["move", ["none", "left", "right"]],
  ["aim", ["none", "up", "down"]],
  ["fire", ["no", "yes"]],
  ["jump", ["no", "yes"]],
  ["dig", ["no", "yes"]],
  ["rope", ["none", "throw", "release"]],
  ["ropeLength", ["none", "shorter", "longer"]],
  ["weapon", ["none", "next", "previous"]],
];

export const ACTION_SIZES = ACTION_HEADS.map(([, choices]) => choices.length);

const MOVE_KEYS = [0, KEYS.left, KEYS.right];
const AIM_KEYS = [0, KEYS.aimUp, KEYS.aimDown];
const ROPE_CHOICES = [ROPE.none, ROPE.throw, ROPE.release];
const ROPE_LENGTH_KEYS = [0, KEYS.ropeShorter, KEYS.ropeLonger];
const WEAPON_CHOICES = [0, 1, -1];

/**
 * One worm's action, read out of a flat array of head choices — the shape a
 * batch of them arrives in from a trainer.
 */
export function actionFromHeads(heads, at = 0) {
  return {
    keys:
      MOVE_KEYS[heads[at]] |
      AIM_KEYS[heads[at + 1]] |
      (heads[at + 2] ? KEYS.fire : 0) |
      (heads[at + 3] ? KEYS.jump : 0) |
      (heads[at + 4] ? KEYS.dig : 0) |
      (ROPE_LENGTH_KEYS[heads[at + 6]] ?? 0),
    rope: ROPE_CHOICES[heads[at + 5]] ?? ROPE.none,
    weapon: WEAPON_CHOICES[heads[at + 7]] ?? 0,
  };
}

/**
 * Hand one action to one worm, in the order a room does it: the key bitmask is
 * state the next `world.update()` reads, while the rope and weapon messages
 * take effect the moment they arrive.
 */
export function applyAction(world, worm, action) {
  return applyNormalizedAction(world, worm, normalizeAction(action));
}

/** The same, for a caller that has already normalized and does it every tick. */
export function applyNormalizedAction(world, worm, { keys, rope, weapon }) {
  worm.Wa = keys;
  if (rope === ROPE.throw) worm.kx(world);
  else if (rope === ROPE.release) worm.Pw();
  if (weapon) worm.oq(worm.Ka + weapon);
  return { keys, rope, weapon };
}
