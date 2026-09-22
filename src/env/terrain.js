// Reading the terrain the way the engine does.
//
// These are the same questions `snapshotV20` answers for the dashboard, in a
// form an observation encoder can call a few thousand times per second: plain
// functions over a terrain record, so the hot path allocates nothing. The
// adapter keeps its own inlined copy because it is stringified into the page,
// where module scope does not exist; `test/env-observation.test.js` runs both
// against the same fixture so the two cannot drift apart unnoticed.

// A material is a byte of flags, not a colour: levels each pick their own
// palette, so an index range says nothing.
export const BACKGROUND = 8; // somewhere a worm may be
export const SHOT_STOPS = 4; // rock: a shot ends here
export const DIGGABLE = 3; // dirt: a shot or the dig key goes through it

/** The terrain record every function here takes. */
export function terrainOf({ data, width, height, materialFlags }) {
  return { data, width, height, materialFlags };
}

function flagsAt(terrain, x, y) {
  if (x < 0 || y < 0 || x >= terrain.width || y >= terrain.height) {
    // The boundary is the hardest wall there is: it stops a worm and a shot.
    return SHOT_STOPS;
  }
  return terrain.materialFlags[terrain.data[y * terrain.width + x]];
}

/** Can a worm be here? */
export function solidAt(terrain, x, y) {
  return (flagsAt(terrain, x, y) & BACKGROUND) === 0;
}

/** Terrain no weapon removes, which is what a route has to go around. */
export function rockAt(terrain, x, y) {
  const flags = flagsAt(terrain, x, y);
  return (flags & BACKGROUND) === 0 && (flags & DIGGABLE) === 0;
}

/** Dirt: solid now, gone after a few seconds of digging or shooting. */
export function diggableAt(terrain, x, y) {
  const flags = flagsAt(terrain, x, y);
  return (flags & BACKGROUND) === 0 && (flags & DIGGABLE) !== 0;
}

/** Bits 0-2: the test the rope itself makes, read off `zc.update` in the bundle. */
export const ROPE_HOLDS = 7;

/**
 * Whether a thrown rope stops here.
 *
 * Not the same question as `solidAt`. The rope attaches where any of bits 0-2
 * is set — dirt or rock — and flies on through anything else, while the worm
 * is stopped by anything that is not background. This mod's palette has 175
 * colours with no flags at all and nine with only bit 5, and five of the
 * community maps draw much of their walls in them: a worm walks into such a
 * wall and a rope goes straight through it to whatever is behind. Off the map
 * counts as holding, which is what the engine does at the edge.
 */
export function ropeHoldsAt(terrain, x, y) {
  return (flagsAt(terrain, x, y) & ROPE_HOLDS) !== 0;
}

/**
 * The four kinds of ground an observation tells apart, by material flags.
 *
 * `ghost` is the fourth: solid to the worm, nothing to the rope. It used to be
 * folded into rock, which showed the policy a wall it could hook and could not.
 * The codes are what the patch keeps in its two low bits, so they are fixed —
 * a recording made before `ghost` existed still reads the same.
 */
export const KIND = { rock: 0, dirt: 1, free: 2, ghost: 3 };

export function kindOf(flags) {
  if (flags & BACKGROUND) return KIND.free;
  if (flags & DIGGABLE) return KIND.dirt;
  if (flags & SHOT_STOPS) return KIND.rock;
  return KIND.ghost;
}

/** The same for a pixel; off the map is rock, the way the boundary behaves. */
export function kindAt(terrain, x, y) {
  return kindOf(flagsAt(terrain, x, y));
}

// The engine's own contact test: movement on an axis is blocked only when two
// of that direction's probes are solid, and they sit one pixel away.
const PROBES = {
  up: [[-1, -4], [0, -4], [1, -4]],
  right: [[1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3]],
  down: [[-1, 4], [0, 4], [1, 4]],
  left: [[-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3]],
};
export const BLOCKING_CONTACTS = 2;
export const MAX_CONTACTS = PROBES.right.length;

/** How many of the engine's own probes are solid on each side. */
export function contactsAt(terrain, x, y, into = {}) {
  const px = Math.round(x);
  const py = Math.round(y);
  for (const side of ["up", "right", "down", "left"]) {
    let solid = 0;
    for (const [dx, dy] of PROBES[side]) {
      if (solidAt(terrain, px + dx, py + dy)) solid++;
    }
    into[side] = solid;
  }
  // The engine lifts a worm one pixel per tick when it has headroom, ground
  // under it and something against its side, which is how a worm gets over a
  // small bump without being told to.
  into.stepping =
    into.up < BLOCKING_CONTACTS &&
    into.down > 0 &&
    (into.left > 0 || into.right > 0);
  return into;
}

// Can the worm take a step that way? Measured across the body's own height
// rather than from a grid, because a difference of four pixels decides it.
const WORM_HALF_HEIGHT_PX = 3;
const STEP_AHEAD_PX = 7;
const STEP_UP_PX = 7;

/** One of clear, step, dirt or rock, in that order of preference. */
export const WALK = ["clear", "step", "dirt", "rock"];

function columnBlock(terrain, x, top, bottom) {
  let blocked = false;
  let rock = false;
  for (let y = top; y <= bottom; y++) {
    if (!solidAt(terrain, x, y)) continue;
    blocked = true;
    if (rockAt(terrain, x, y)) rock = true;
  }
  return { blocked, rock };
}

/** What a step to `direction` (-1 left, 1 right) runs into. */
export function walkProbe(terrain, x, y, direction) {
  const px = Math.round(x) + direction * STEP_AHEAD_PX;
  const top = Math.round(y) - WORM_HALF_HEIGHT_PX;
  const bottom = Math.round(y) + WORM_HALF_HEIGHT_PX;
  const ahead = columnBlock(terrain, px, top, bottom);
  if (!ahead.blocked) return "clear";
  if (!columnBlock(terrain, px, top - STEP_UP_PX, bottom - STEP_UP_PX).blocked)
    return "step";
  // Dirt is a door that takes a few seconds to open; rock is a wall.
  return ahead.rock ? "rock" : "dirt";
}

/**
 * Pixels from (x, y) to the first solid pixel along a direction, or `null` when
 * nothing is within `limit`. Distances, not pictures, are what a rope and a gun
 * are actually decided on.
 */
export function reach(terrain, x, y, dx, dy, limit) {
  for (let step = 1; step <= limit; step++) {
    const px = Math.round(x + dx * step);
    const py = Math.round(y + dy * step);
    if (px < 0 || py < 0 || px >= terrain.width || py >= terrain.height) return null;
    if (solidAt(terrain, px, py)) return step;
  }
  return null;
}
