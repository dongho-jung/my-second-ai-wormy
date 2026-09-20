// Turning an action back into the keys a person would hold.
//
// The headless environment writes the input bitmask straight onto the worm. A
// live game cannot: it reads the keyboard, and the mapping from held keys to
// that bitmask is not one bit per key. Read off `Kc.ea()` in the v20 bundle:
//
//   Up            -> 4,   but 4|64  while ChangeWeap is held
//   Down          -> 8,   but 8|128 while ChangeWeap is held
//   Left, Right   -> 1, 2 — and holding BOTH also sets 256, which is dig
//   Jump          -> 32,  but only while ChangeWeap is not held
//   Left/Right    -> ignored for movement while ChangeWeap is held; they
//                    change weapon instead
//   ShortenRope   -> 64
//   LengthenRope  -> 128
//   Dig           -> 256
//   Fire          -> 16
//
// So an action that a policy can express is not automatically an action a
// player can press, and the two ways of reaching 256 are the kind of overlap
// that quietly breaks a transfer. `bitsFromKeys` below is that function written
// out again, and a test runs every bitmask the policy can emit through
// `keysForBits` and back to prove the round trip is exact.
//
// Rope throw and weapon change are not in the bitmask at all — they are their
// own messages — so they are their own keys here too.

/**
 * The keys these actions are on, as `KeyboardEvent.code` — the form the game
 * stores in `localStorage["player_keys"]` and the form a synthesised event has
 * to carry. This is the layout the player has set up in Settings > Input.
 *
 * ChangeWeap is deliberately on nothing. It is the modifier that turns aim into
 * rope length and stops the worm walking, and leaving it unbound means no
 * sequence of presses can trigger it by accident.
 */
export const BINDINGS = {
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
  KeyD: "Fire",
  KeyS: "Jump",
  KeyQ: "NextWeap",
  KeyW: "PrevWeap",
  KeyR: "Reload",
  KeyC: "Dig",
  KeyA: "NinjaRope",
  ShiftLeft: "ShortenRope",
  // LengthenRope is deliberately on nothing, the same as ChangeWeap: the action
  // space has no head for rope length, so nothing can ask for it.
  Tab: "Scoreboard",
  Enter: "Chat",
};

/** Which key each action is on, the way round a driver needs it. */
export function codesByAction(bindings = BINDINGS) {
  const codes = {};
  for (const [code, action] of Object.entries(bindings)) {
    if (!(action in codes)) codes[action] = code;
  }
  return codes;
}

/**
 * The actions a driver has to be able to press, and what is missing from a
 * given binding table. Rope length is not here: the policy never asks for it.
 */
export const REQUIRED_ACTIONS = [
  "Up",
  "Down",
  "Left",
  "Right",
  "Fire",
  "Jump",
  "Dig",
  "NinjaRope",
  "NextWeap",
  "PrevWeap",
];

export function missingBindings(bindings) {
  const codes = codesByAction(bindings);
  return REQUIRED_ACTIONS.filter((action) => !codes[action]);
}

/** The action names the game binds keys to, as its own settings spell them. */
export const ACTIONS = {
  up: "Up",
  down: "Down",
  left: "Left",
  right: "Right",
  fire: "Fire",
  jump: "Jump",
  dig: "Dig",
  changeWeapon: "ChangeWeap",
  rope: "NinjaRope",
  nextWeapon: "NextWeap",
  previousWeapon: "PrevWeap",
  shortenRope: "ShortenRope",
  lengthenRope: "LengthenRope",
  reload: "Reload",
};

export const KEY_BITS = {
  left: 1,
  right: 2,
  aimUp: 4,
  aimDown: 8,
  fire: 16,
  jump: 32,
  ropeShorter: 64,
  ropeLonger: 128,
  dig: 256,
};

/**
 * The bitmask a set of held actions produces, exactly as the live game builds
 * it. Kept here so the inverse below can be checked against it rather than
 * trusted.
 */
export function bitsFromKeys(held) {
  const down = (name) => held.has(ACTIONS[name] ?? name);
  let bits = 0;
  const changing = down("changeWeapon");
  if (down("up")) bits = changing ? 68 : 4;
  if (down("down")) bits |= changing ? 8 | 128 : 8;
  if (!changing) {
    const left = down("left");
    const right = down("right");
    if (left) bits |= 1;
    if (right) bits |= 2;
    // Both at once is how a player digs without a dig key bound.
    if (left && right) bits |= 256;
    if (down("jump")) bits |= 32;
  }
  if (down("shortenRope")) bits |= 64;
  if (down("lengthenRope")) bits |= 128;
  if (down("dig")) bits |= 256;
  if (down("fire")) bits |= 16;
  return bits;
}

/**
 * The actions to hold for a given bitmask.
 *
 * Every bit has a key of its own, so nothing needs the ChangeWeap modifier and
 * none of its side effects can happen by accident. The one bit a policy cannot
 * ask for is left and right together, which the engine reads as dig — and the
 * action space already treats them as one three-way choice, so it never does.
 */
export function keysForBits(bits) {
  if (bits & KEY_BITS.left && bits & KEY_BITS.right) {
    throw new Error(
      "left and right together is dig, not standing still: the action space has to keep them apart",
    );
  }
  const held = new Set();
  const hold = (name) => held.add(ACTIONS[name]);
  if (bits & KEY_BITS.left) hold("left");
  if (bits & KEY_BITS.right) hold("right");
  if (bits & KEY_BITS.aimUp) hold("up");
  if (bits & KEY_BITS.aimDown) hold("down");
  if (bits & KEY_BITS.fire) hold("fire");
  if (bits & KEY_BITS.jump) hold("jump");
  if (bits & KEY_BITS.ropeShorter) hold("shortenRope");
  if (bits & KEY_BITS.ropeLonger) hold("lengthenRope");
  if (bits & KEY_BITS.dig) hold("dig");
  return held;
}

/** What to press once, rather than hold, for the messages that are not bits. */
export function pressesForAction({ rope = 0, weapon = 0 } = {}) {
  const presses = [];
  if (rope !== 0) presses.push(ACTIONS.rope);
  if (weapon > 0) presses.push(ACTIONS.nextWeapon);
  else if (weapon < 0) presses.push(ACTIONS.previousWeapon);
  return presses;
}

/**
 * A held bitmask read back as the seven choices a policy emits.
 *
 * This is how a person's play becomes something to learn from: the game
 * replicates every player's input to everyone in the room, so what somebody
 * pressed arrives in the same field the policy writes.
 *
 * Left and right together is the engine's other way of digging, so it is read
 * as dig with no movement rather than as a contradiction.
 */
export function headsFromKeys(keys, { rope = 0, weapon = 0 } = {}) {
  const left = Boolean(keys & KEY_BITS.left);
  const right = Boolean(keys & KEY_BITS.right);
  const both = left && right;
  return Uint8Array.from([
    both ? 0 : left ? 1 : right ? 2 : 0,
    keys & KEY_BITS.aimUp ? 1 : keys & KEY_BITS.aimDown ? 2 : 0,
    keys & KEY_BITS.fire ? 1 : 0,
    keys & KEY_BITS.jump ? 1 : 0,
    both || keys & KEY_BITS.dig ? 1 : 0,
    rope > 0 ? 1 : rope < 0 ? 2 : 0,
    weapon > 0 ? 1 : weapon < 0 ? 2 : 0,
  ]);
}
