// What a policy is given each step.
//
// Two pictures of the same moment, because neither one alone is enough. The
// vector is cheap and carries everything a number can say — health, ammo, where
// the other worm is, how far the walls are — but a person has to decide in
// advance what to measure, and a staircase and a sealed wall can end up as the
// same two distances. The patch is the terrain itself, small and close, and the
// network works out for itself what a ledge is.
//
// So: state in the vector, shape in the patch. Start a task on the vector alone
// to prove the pipeline, add the patch when the terrain starts to matter.
import {
  BACKGROUND,
  DIGGABLE,
  MAX_CONTACTS,
  WALK,
  contactsAt,
  solidAt,
  walkProbe,
} from "./terrain.js";

/** How far a ray looks before it reports nothing there. */
export const RAY_RANGE_PX = 120;
export const RAY_COUNT = 16;
const RAY_STEP_PX = 1;

// Rays start pointing right and go round clockwise on screen, where y grows
// downward: 0 right, 4 down, 8 left, 12 up.
const RAY_DIRECTIONS = Array.from({ length: RAY_COUNT }, (_, k) => {
  const angle = (k / RAY_COUNT) * Math.PI * 2;
  return [Math.cos(angle), Math.sin(angle)];
});

// The distance everything relative is measured against: the rope's reach, which
// is roughly as far as a worm can act in one move.
const REACH_PX = 300;
const HEALTH_MAX = 100;
const WEAPON_SLOTS = 5;
/** Only the closest few shots go in the vector; the patch carries the rest. */
export const PROJECTILE_SLOTS = 3;

/**
 * The vector, field by field. Exported because a layout you cannot print is a
 * layout you cannot debug: `VECTOR_OFFSETS.foe` is where the other worm starts.
 */
export const VECTOR_LAYOUT = [
  ["rays", RAY_COUNT], //   distance to the first solid pixel, 1 = clear to the limit
  ["health", 1],
  ["velocity", 2], //       px per tick, already around 1
  ["aim", 2], //            cos and sin, so the wrap at PI is not a cliff
  ["facing", 1], //         -1 left, +1 right
  ["contacts", 4], //       the engine's own up/right/down/left probe counts
  ["stepping", 1], //       the engine is lifting the worm over a bump
  ["walkLeft", WALK.length], //   one-hot clear/step/dirt/rock
  ["walkRight", WALK.length],
  ["weapons", WEAPON_SLOTS * 3], // per slot: ammo left, ready to fire, selected
  ["rope", 5], //           out, attached, where it is, how long
  ["foe", 9], //            alive, where, how far, which way, health, speed
  ["projectiles", PROJECTILE_SLOTS * 4], // the nearest shots: where and where to
];

export const VECTOR_OFFSETS = {};
export const VECTOR_SIZE = VECTOR_LAYOUT.reduce((at, [name, size]) => {
  VECTOR_OFFSETS[name] = at;
  return at + size;
}, 0);

/** The patch: a small square of terrain centred on the worm. */
export const PATCH = {
  // Free space gets a channel of its own instead of being the absence of the
  // other two, so "solid" and "nothing measured" never look alike. Past the
  // edge of the map reads as rock, which is exactly how it behaves.
  channels: ["rock", "dirt", "free", "projectile"],
  cells: 32,
  // Two pixels per cell: 64 px across, about nine worm heights, which is the
  // range footwork happens in. The rays cover the rest.
  scalePx: 2,
};
export const PATCH_SIZE = PATCH.channels.length * PATCH.cells * PATCH.cells;

const clamp = (value, low, high) => (value < low ? low : value > high ? high : value);

/** The pictures a policy is given, and their order of expense. */
export const OBSERVATIONS = ["vector", "patch"];

/**
 * Both pictures, or only the ones asked for. The patch costs about ten times
 * the vector, so a task that does not turn on the terrain — walking to a point,
 * a first pipeline check — should ask for `["vector"]` and get the speed back.
 */
export function observe(view, into = {}, kinds = OBSERVATIONS) {
  const out = {};
  if (kinds.includes("vector")) out.vector = encodeVector(view, into.vector);
  if (kinds.includes("patch")) out.patch = encodePatch(view, into.patch);
  return out;
}

const contacts = {};

export function encodeVector(view, into = new Float32Array(VECTOR_SIZE)) {
  into.fill(0);
  const { self, terrain } = view;
  if (!self.alive) return into; // a dead worm sees nothing until it respawns
  const { x, y } = self.position;
  let at = 0;

  for (const [dx, dy] of RAY_DIRECTIONS) {
    let distance = 0;
    while (
      distance < RAY_RANGE_PX &&
      !solidAt(terrain, Math.round(x + dx * distance), Math.round(y + dy * distance))
    ) {
      distance += RAY_STEP_PX;
    }
    into[at++] = distance / RAY_RANGE_PX;
  }

  into[at++] = self.health / HEALTH_MAX;
  into[at++] = self.velocity.x;
  into[at++] = self.velocity.y;
  into[at++] = Math.cos(self.aimRadians);
  into[at++] = Math.sin(self.aimRadians);
  into[at++] = self.facing === "right" ? 1 : -1;

  contactsAt(terrain, x, y, contacts);
  into[at++] = contacts.up / MAX_CONTACTS;
  into[at++] = contacts.right / MAX_CONTACTS;
  into[at++] = contacts.down / MAX_CONTACTS;
  into[at++] = contacts.left / MAX_CONTACTS;
  into[at++] = contacts.stepping ? 1 : 0;
  into[at + WALK.indexOf(walkProbe(terrain, x, y, -1))] = 1;
  at += WALK.length;
  into[at + WALK.indexOf(walkProbe(terrain, x, y, 1))] = 1;
  at += WALK.length;

  for (let slot = 0; slot < WEAPON_SLOTS; slot++) {
    const weapon = self.weapons[slot];
    if (!weapon) {
      at += 3;
      continue;
    }
    into[at++] = weapon.capacity > 0 ? weapon.ammo / weapon.capacity : 0;
    // One number for "may I fire right now", which is what the decision is.
    into[at++] =
      weapon.ammo > 0 && weapon.cooldownTicksRemaining <= 0 ? 1 : 0;
    into[at++] = slot === self.selectedWeapon ? 1 : 0;
  }

  if (self.rope) {
    into[at++] = 1;
    into[at++] = self.rope.attached ? 1 : 0;
    into[at++] = clamp((self.rope.position.x - x) / REACH_PX, -1, 1);
    into[at++] = clamp((self.rope.position.y - y) / REACH_PX, -1, 1);
    into[at++] = clamp(self.rope.length / REACH_PX, 0, 1);
  } else {
    at += 5;
  }

  const foe = nearestFoe(view);
  if (foe) {
    const dx = foe.position.x - x;
    const dy = foe.position.y - y;
    const distance = Math.hypot(dx, dy) || 1;
    into[at++] = 1;
    into[at++] = clamp(dx / REACH_PX, -1, 1);
    into[at++] = clamp(dy / REACH_PX, -1, 1);
    into[at++] = clamp(distance / REACH_PX, 0, 1);
    // Direction survives even when the distance has been clipped.
    into[at++] = dx / distance;
    into[at++] = dy / distance;
    into[at++] = foe.health / HEALTH_MAX;
    into[at++] = foe.velocity.x;
    into[at++] = foe.velocity.y;
  } else {
    at += 9;
  }

  // Own shots are in here too: a worm's own explosion is most of the damage it
  // ever takes, so there is nothing to gain by hiding them.
  for (const shot of nearestProjectiles(view)) {
    into[at++] = clamp((shot.position.x - x) / REACH_PX, -1, 1);
    into[at++] = clamp((shot.position.y - y) / REACH_PX, -1, 1);
    into[at++] = shot.velocity.x;
    into[at++] = shot.velocity.y;
  }
  return into;
}

/** The top-left pixel of the patch, which every cell is counted from. */
export function patchOriginOf(view) {
  const half = (PATCH.cells >> 1) * PATCH.scalePx;
  return {
    x: Math.round(view.self.position.x) - half,
    y: Math.round(view.self.position.y) - half,
  };
}

/** Which cell a pixel falls in, or `null` when it is outside the patch. */
export function patchCellOf(origin, x, y) {
  const column = Math.floor((x - origin.x) / PATCH.scalePx);
  const row = Math.floor((y - origin.y) / PATCH.scalePx);
  if (column < 0 || row < 0 || column >= PATCH.cells || row >= PATCH.cells)
    return null;
  return { column, row, cell: row * PATCH.cells + column };
}

// Scratch, reused between calls. The level row each of the patch's pixel rows
// lands on, and the level x of each of its pixel columns, with -1 for the ones
// that fall off the map.
const PIXEL_ROWS = new Int32Array(PATCH.cells * PATCH.scalePx);
const PIXEL_COLUMNS = new Int32Array(PATCH.cells * PATCH.scalePx);
// The channel order is the hardness order, so the lowest code wins and is also
// the channel it is written to.
const ROCK = 0;
const DIRT = 1;
const FREE = 2;

export function encodePatch(view, into = new Float32Array(PATCH_SIZE)) {
  into.fill(0);
  const { self, terrain } = view;
  if (!self.alive) return into;
  const { cells, scalePx } = PATCH;
  const plane = cells * cells;
  const { data, width, height, materialFlags } = terrain;
  const origin = patchOriginOf(view);
  // The bounds arithmetic is done once for the whole patch rather than once per
  // pixel: 64 rows and 64 columns against 4,096 cells.
  for (let pixel = 0; pixel < cells * scalePx; pixel++) {
    const y = origin.y + pixel;
    PIXEL_ROWS[pixel] = y < 0 || y >= height ? -1 : y * width;
    const x = origin.x + pixel;
    PIXEL_COLUMNS[pixel] = x < 0 || x >= width ? -1 : x;
  }
  for (let row = 0; row < cells; row++) {
    for (let column = 0; column < cells; column++) {
      // A cell stands for `scalePx` squared real pixels and answers for the
      // hardest of them, so a wall one pixel thick cannot fall between two
      // samples and be reported as open air. Off the level answers rock, which
      // is exactly how the boundary behaves.
      let hardest = FREE;
      for (let down = 0; down < scalePx && hardest !== ROCK; down++) {
        const levelRow = PIXEL_ROWS[row * scalePx + down];
        if (levelRow < 0) {
          hardest = ROCK;
          break;
        }
        for (let across = 0; across < scalePx; across++) {
          const x = PIXEL_COLUMNS[column * scalePx + across];
          if (x < 0) {
            hardest = ROCK;
            break;
          }
          const flags = materialFlags[data[levelRow + x]];
          const kind = flags & BACKGROUND ? FREE : flags & DIGGABLE ? DIRT : ROCK;
          if (kind < hardest) hardest = kind;
          if (hardest === ROCK) break;
        }
      }
      into[hardest * plane + row * cells + column] = 1;
    }
  }
  // Shots go in a channel of their own, so a policy handles ten of them and one
  // of them with the same weights.
  const shots = 3 * plane;
  for (const shot of view.projectiles) {
    const at = patchCellOf(origin, shot.position.x, shot.position.y);
    if (at) into[shots + at.cell] = 1;
  }
  return into;
}

/** The living foe a policy should be looking at: the closest one. */
export function nearestFoe(view) {
  const { self } = view;
  if (!self.alive) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const foe of view.foes) {
    if (!foe.alive) continue;
    const distance =
      (foe.position.x - self.position.x) ** 2 +
      (foe.position.y - self.position.y) ** 2;
    if (distance < bestDistance) {
      best = foe;
      bestDistance = distance;
    }
  }
  return best;
}

const nearest = [];

/** The `PROJECTILE_SLOTS` closest shots, closest first. */
export function nearestProjectiles(view) {
  nearest.length = 0;
  const { x, y } = view.self.position;
  for (const shot of view.projectiles) {
    const distance = (shot.position.x - x) ** 2 + (shot.position.y - y) ** 2;
    let at = nearest.length;
    while (at > 0 && nearest[at - 1].distance > distance) at--;
    if (at >= PROJECTILE_SLOTS) continue;
    nearest.splice(at, 0, { shot, distance });
    if (nearest.length > PROJECTILE_SLOTS) nearest.pop();
  }
  return nearest.map((entry) => entry.shot);
}
