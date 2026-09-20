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
import { WEAPON_FEATURE_COUNT } from "./engine.js";
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
 * How many other worms the vector describes, by default. Two, because a
 * free-for-all is not a duel: the one shooting at you and the one you are
 * shooting at are often not the same worm, and a policy that can only see one
 * of them cannot choose.
 *
 * It is a setting, not a constant, because the table is not always three. A
 * vector built for four foes and used in a duel simply leaves three of them
 * zeroed — which is how one policy can play 1v1 and a five-way without being
 * retrained, at the cost of a few dozen numbers it will learn to ignore.
 */
export const DEFAULT_FOE_SLOTS = 2;
const FOE_FIELDS = 9;
/** Health and weapon crates worth walking to. */
export const PICKUP_SLOTS = 3;
const PICKUP_FIELDS = 4;
/** Of a shot's weapon, the part that says how afraid to be. */
const SHOT_WEAPON_FIELDS = 4;

/**
 * The vector, field by field. Exported because a layout you cannot print is a
 * layout you cannot debug: `VECTOR_OFFSETS.foe` is where the other worm starts.
 */
/**
 * The shape of one vector: what is in it, in order, and where each field starts.
 * Exported because a layout you cannot print is a layout you cannot debug —
 * `spec.offsets.foes` is where the other worms start.
 */
export function observationSpec({
  foeSlots = DEFAULT_FOE_SLOTS,
  projectileSlots = PROJECTILE_SLOTS,
  pickupSlots = PICKUP_SLOTS,
  weaponFeatures = null,
} = {}) {
  const layout = [
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
    ["weaponsReload", WEAPON_SLOTS], // how far through reloading each one is
    ["weapon", WEAPON_FEATURE_COUNT], // what the held weapon actually does
    ["rope", 5], //           out, attached, where it is, how long
    // Each foe: alive, where, how far, which way, health, speed — and what it
    // is pointing at you, which decides whether to close or break off.
    ["foes", foeSlots * (FOE_FIELDS + WEAPON_FEATURE_COUNT)],
    // The nearest shots: where, where to, and how much they are going to hurt.
    ["projectiles", projectileSlots * (4 + SHOT_WEAPON_FIELDS)],
    ["pickups", pickupSlots * PICKUP_FIELDS], // health and weapon crates nearby
  ];
  const offsets = {};
  const vectorSize = layout.reduce((at, [name, size]) => {
    offsets[name] = at;
    return at + size;
  }, 0);
  return {
    foeSlots,
    projectileSlots,
    pickupSlots,
    weaponFeatures,
    layout,
    offsets,
    vectorSize,
  };
}

export const DEFAULT_SPEC = observationSpec();
export const VECTOR_LAYOUT = DEFAULT_SPEC.layout;
export const VECTOR_OFFSETS = DEFAULT_SPEC.offsets;
export const VECTOR_SIZE = DEFAULT_SPEC.vectorSize;

/**
 * The patch: what the player can see, and no more.
 *
 * The game draws a 426x240 window of the level around the worm — in the bundle,
 * `canvas.width=426; canvas.height=240` and a camera held at `+213/+120`. That
 * window is the whole of what a person at the keyboard has to go on, so it is
 * what the policy gets too. An earlier version of this showed 64x64 px, nine
 * worm-heights, and the worms behaved accordingly: nothing within sight said
 * where to go, so they fired the rope at the floor until the round ended.
 *
 * Odd counts so the worm sits in one exact centre cell rather than straddling
 * four. 121 rows spans 242 px against the true 240 — two pixels of margin is
 * worth more than an off-centre worm.
 */
export const PATCH = {
  // Free space gets a channel of its own instead of being the absence of the
  // other two, so "solid" and "nothing measured" never look alike. Past the
  // edge of the map reads as rock, which is exactly how it behaves.
  channels: ["rock", "dirt", "free", "projectile"],
  columns: 213,
  rows: 121,
  // Two pixels per cell, the resolution footwork happens at.
  scalePx: 2,
};
export const PATCH_CELLS = PATCH.columns * PATCH.rows;
export const PATCH_SIZE = PATCH.channels.length * PATCH_CELLS;
/** The shape a convolution reads it as: channels, rows, columns. */
export const PATCH_SHAPE = [PATCH.channels.length, PATCH.rows, PATCH.columns];

/**
 * The whole level, small.
 *
 * The patch above is nine worm-heights across. The map is five hundred pixels
 * across. A policy given only the patch can climb a ledge in front of it and
 * has no way at all to decide which direction the rest of the match is in —
 * which is exactly what it looks like: worms that fire the rope at the floor
 * until the round ends, because nothing in what they can see says where to go.
 *
 * So: the level stretched onto a fixed grid, as three fractions rather than a
 * hardest-wins code. At this scale a cell holds hundreds of pixels and "a
 * quarter of this is open" is the useful thing to know, not "some of it is
 * rock". The fourth channel is where everybody is.
 */
export const MAP = {
  channels: ["free", "dirt", "rock", "occupants"],
  cells: 32,
};
export const MAP_CELLS = MAP.cells * MAP.cells;
export const MAP_SIZE = MAP.channels.length * MAP_CELLS;
/** Where the occupant channel starts, and what it writes. */
const MAP_OCCUPANTS = 3 * MAP_CELLS;
export const MAP_SELF = 255;
export const MAP_FOE = 128;

const clamp = (value, low, high) => (value < low ? low : value > high ? high : value);

/** The pictures a policy is given, and their order of expense. */
export const OBSERVATIONS = ["vector", "patch"];

/**
 * Both pictures, or only the ones asked for. The patch costs about ten times
 * the vector, so a task that does not turn on the terrain — walking to a point,
 * a first pipeline check — should ask for `["vector"]` and get the speed back.
 */
export function observe(
  view,
  into = {},
  kinds = OBSERVATIONS,
  spec = DEFAULT_SPEC,
  mapTerrain = null,
) {
  const out = {};
  if (kinds.includes("vector")) out.vector = encodeVector(view, into.vector, spec);
  if (kinds.includes("patch")) out.patch = encodePatch(view, into.patch);
  if (kinds.includes("patchBytes")) out.patchBytes = encodePatchBytes(view, into.patchBytes);
  if (kinds.includes("map")) {
    out.map = mapTerrain
      ? encodeMap(view, mapTerrain, into.map)
      : encodeMap(view, encodeMapTerrain(view.terrain), into.map);
  }
  return out;
}

const contacts = {};

export function encodeVector(view, into = null, spec = DEFAULT_SPEC) {
  into ??= new Float32Array(spec.vectorSize);
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

  // How far through reloading each slot is: a policy that switches to an empty
  // gun and stands there is one that was never told.
  for (let slot = 0; slot < WEAPON_SLOTS; slot++) {
    const weapon = self.weapons[slot];
    if (!weapon) {
      at++;
      continue;
    }
    const waiting = weapon.reloadTicksRemaining + weapon.cooldownTicksRemaining;
    into[at++] = waiting > 0 ? Math.min(1, waiting / 300) : 0;
  }
  at = writeWeapon(into, at, spec, self.weapons?.[self.selectedWeapon]?.id);

  if (self.rope) {
    into[at++] = 1;
    into[at++] = self.rope.attached ? 1 : 0;
    into[at++] = clamp((self.rope.position.x - x) / REACH_PX, -1, 1);
    into[at++] = clamp((self.rope.position.y - y) / REACH_PX, -1, 1);
    into[at++] = clamp(self.rope.length / REACH_PX, 0, 1);
  } else {
    at += 5;
  }

  const foes = nearestFoes(view, spec.foeSlots);
  for (let slot = 0; slot < spec.foeSlots; slot++) {
    const foe = foes[slot];
    if (!foe) {
      // The whole slot, weapon and all: skipping only the first half shifts
      // every field after the foes block by ten whenever somebody is dead.
      at += FOE_FIELDS + WEAPON_FEATURE_COUNT;
      continue;
    }
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
    at = writeWeapon(into, at, spec, foe.weapons?.[foe.selectedWeapon]?.id);
  }

  // Own shots are in here too: a worm's own explosion is most of the damage it
  // ever takes, so there is nothing to gain by hiding them.
  const shots = nearestProjectiles(view, spec.projectileSlots);
  for (let slot = 0; slot < spec.projectileSlots; slot++) {
    const shot = shots[slot];
    if (!shot) {
      at += 4 + SHOT_WEAPON_FIELDS;
      continue;
    }
    into[at++] = clamp((shot.position.x - x) / REACH_PX, -1, 1);
    into[at++] = clamp((shot.position.y - y) / REACH_PX, -1, 1);
    into[at++] = shot.velocity.x;
    into[at++] = shot.velocity.y;
    // Enough of the weapon to know whether to dodge it or ignore it: what it
    // does on a hit, what it does to whoever fired it, whether it arcs, how
    // fast it is.
    const features = spec.weaponFeatures;
    const from = shot.weaponId == null ? -1 : shot.weaponId * WEAPON_FEATURE_COUNT;
    for (const index of [4, 6, 1, 0]) {
      into[at++] = features && from >= 0 ? features[from + index] : 0;
    }
  }

  // Crates worth a detour, nearest first.
  const crates = nearestPickups(view, spec.pickupSlots);
  for (let slot = 0; slot < spec.pickupSlots; slot++) {
    const crate = crates[slot];
    if (!crate) {
      at += PICKUP_FIELDS;
      continue;
    }
    into[at++] = clamp((crate.position.x - x) / REACH_PX, -1, 1);
    into[at++] = clamp((crate.position.y - y) / REACH_PX, -1, 1);
    into[at++] = crate.kind === "health" ? 1 : 0;
    into[at++] = crate.kind === "health" ? 0 : 1;
  }
  return into;
}

/** One weapon's measured character, or zeros when there is nothing to say. */
function writeWeapon(into, at, spec, weaponId) {
  const features = spec.weaponFeatures;
  const from = weaponId == null ? -1 : weaponId * WEAPON_FEATURE_COUNT;
  for (let index = 0; index < WEAPON_FEATURE_COUNT; index++) {
    into[at + index] = features && from >= 0 ? features[from + index] : 0;
  }
  return at + WEAPON_FEATURE_COUNT;
}

const crates = [];

/** The nearest crates, closest first. */
export function nearestPickups(view, count = PICKUP_SLOTS) {
  crates.length = 0;
  if (!view.self.alive || !view.pickups?.length) return crates;
  const { x, y } = view.self.position;
  for (const crate of view.pickups) {
    const distance = (crate.position.x - x) ** 2 + (crate.position.y - y) ** 2;
    let at = crates.length;
    while (at > 0 && crates[at - 1].distance > distance) at--;
    if (at >= count) continue;
    crates.splice(at, 0, { crate, distance });
    if (crates.length > count) crates.pop();
  }
  return crates.map((entry) => entry.crate);
}

/**
 * The level's terrain on the map grid, as three fractions per cell. Slow —
 * it reads every pixel — so a caller keeps the result and refreshes it on a
 * clock of its own rather than every decision.
 */
export function encodeMapTerrain(terrain, into = new Uint8Array(3 * MAP_CELLS)) {
  into.fill(0);
  const { data, width, height, materialFlags } = terrain;
  const { cells } = MAP;
  const counts = new Uint32Array(3 * MAP_CELLS);
  const perCell = new Uint32Array(MAP_CELLS);
  for (let y = 0; y < height; y++) {
    const row = Math.min(cells - 1, ((y * cells) / height) | 0);
    const base = y * width;
    for (let x = 0; x < width; x++) {
      const cell = row * cells + Math.min(cells - 1, ((x * cells) / width) | 0);
      const flags = materialFlags[data[base + x]];
      const kind = flags & BACKGROUND ? 0 : flags & DIGGABLE ? 1 : 2;
      counts[kind * MAP_CELLS + cell]++;
      perCell[cell]++;
    }
  }
  for (let cell = 0; cell < MAP_CELLS; cell++) {
    const total = perCell[cell] || 1;
    for (let kind = 0; kind < 3; kind++) {
      into[kind * MAP_CELLS + cell] = Math.round((counts[kind * MAP_CELLS + cell] / total) * 255);
    }
  }
  return into;
}

/** Where this worm and the others are on that grid. */
export function encodeMapOccupants(view, into = new Uint8Array(MAP_CELLS)) {
  into.fill(0);
  const { terrain, self } = view;
  const { cells } = MAP;
  const at = (position) => {
    const column = Math.min(cells - 1, Math.max(0, ((position.x * cells) / terrain.width) | 0));
    const row = Math.min(cells - 1, Math.max(0, ((position.y * cells) / terrain.height) | 0));
    return row * cells + column;
  };
  for (const foe of view.foes) if (foe.alive) into[at(foe.position)] = MAP_FOE;
  if (self.alive) into[at(self.position)] = MAP_SELF;
  return into;
}

/** The two halves together, in the layout the trainer reads. */
export function encodeMap(view, terrainBytes, into = new Uint8Array(MAP_SIZE)) {
  into.set(terrainBytes.subarray(0, MAP_OCCUPANTS), 0);
  encodeMapOccupants(view, into.subarray(MAP_OCCUPANTS, MAP_OCCUPANTS + MAP_CELLS));
  return into;
}

/** The top-left pixel of the patch, which every cell is counted from. */
export function patchOriginOf(view) {
  const { columns, rows, scalePx } = PATCH;
  const half = scalePx >> 1;
  return {
    x: Math.round(view.self.position.x) - ((columns >> 1) * scalePx + half),
    y: Math.round(view.self.position.y) - ((rows >> 1) * scalePx + half),
  };
}

/** Which cell a pixel falls in, or `null` when it is outside the patch. */
export function patchCellOf(origin, x, y) {
  const column = Math.floor((x - origin.x) / PATCH.scalePx);
  const row = Math.floor((y - origin.y) / PATCH.scalePx);
  if (column < 0 || row < 0 || column >= PATCH.columns || row >= PATCH.rows)
    return null;
  return { column, row, cell: row * PATCH.columns + column };
}

// Scratch, reused between calls. The level row each of the patch's pixel rows
// lands on, and the level x of each of its pixel columns, with -1 for the ones
// that fall off the map.
const PIXEL_ROWS = new Int32Array(PATCH.rows * PATCH.scalePx);
const PIXEL_COLUMNS = new Int32Array(PATCH.columns * PATCH.scalePx);
// The first three channels are the hardness order, so the lowest code wins and
// is also the channel it expands to.
const ROCK = 0;
const DIRT = 1;
const FREE = 2;
/** Bit set on a cell that has a shot in it, alongside the terrain code. */
export const PATCH_PROJECTILE = 4;
export const PATCH_KIND = 3;

const patchScratch = new Uint8Array(PATCH_CELLS);

/**
 * The patch as one byte per cell — the form it is stored and sent in.
 *
 * Bits 0-1 are the terrain (0 rock, 1 dirt, 2 free) and bit 2 says a shot is in
 * the cell. A quarter of the size of the one-hot form and the same information:
 * a trainer expands it on the GPU, where the expansion is free, and a run that
 * ships a million of these across a pipe ships a quarter of the bytes.
 */
export function encodePatchBytes(view, into = new Uint8Array(PATCH_CELLS)) {
  into.fill(0);
  const { self, terrain } = view;
  if (!self.alive) return into;
  const { columns, rows, scalePx } = PATCH;
  const { data, width, height, materialFlags } = terrain;
  const origin = patchOriginOf(view);
  // The bounds arithmetic is done once for the whole patch rather than once per
  // pixel: 242 rows and 426 columns against 25,773 cells.
  for (let pixel = 0; pixel < rows * scalePx; pixel++) {
    const y = origin.y + pixel;
    PIXEL_ROWS[pixel] = y < 0 || y >= height ? -1 : y * width;
  }
  for (let pixel = 0; pixel < columns * scalePx; pixel++) {
    const x = origin.x + pixel;
    PIXEL_COLUMNS[pixel] = x < 0 || x >= width ? -1 : x;
  }
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
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
      into[row * columns + column] = hardest;
    }
  }
  // Shots get a bit of their own, so a policy handles ten of them and one of
  // them with the same weights.
  for (const shot of view.projectiles) {
    const at = patchCellOf(origin, shot.position.x, shot.position.y);
    if (at) into[at.cell] |= PATCH_PROJECTILE;
  }
  return into;
}

/** The same patch as one-hot float planes, for a consumer that wants them. */
export function encodePatch(view, into = new Float32Array(PATCH_SIZE)) {
  into.fill(0);
  if (!view.self.alive) return into;
  const bytes = encodePatchBytes(view, patchScratch);
  const plane = PATCH_CELLS;
  for (let cell = 0; cell < plane; cell++) {
    const byte = bytes[cell];
    into[(byte & PATCH_KIND) * plane + cell] = 1;
    if (byte & PATCH_PROJECTILE) into[3 * plane + cell] = 1;
  }
  return into;
}

const closest = [];

/** The living foes a policy should be looking at, closest first. */
export function nearestFoes(view, count = DEFAULT_FOE_SLOTS) {
  closest.length = 0;
  const { self } = view;
  if (!self.alive) return closest;
  for (const foe of view.foes) {
    if (!foe.alive) continue;
    const distance =
      (foe.position.x - self.position.x) ** 2 +
      (foe.position.y - self.position.y) ** 2;
    let at = closest.length;
    while (at > 0 && closest[at - 1].distance > distance) at--;
    if (at >= count) continue;
    closest.splice(at, 0, { foe, distance });
    if (closest.length > count) closest.pop();
  }
  return closest.map((entry) => entry.foe);
}

/** The single closest living foe, or null. */
export function nearestFoe(view) {
  return nearestFoes(view, 1)[0] ?? null;
}

const nearest = [];

/** The closest shots, closest first. */
export function nearestProjectiles(view, count = PROJECTILE_SLOTS) {
  nearest.length = 0;
  const { x, y } = view.self.position;
  for (const shot of view.projectiles) {
    const distance = (shot.position.x - x) ** 2 + (shot.position.y - y) ** 2;
    let at = nearest.length;
    while (at > 0 && nearest[at - 1].distance > distance) at--;
    if (at >= count) continue;
    nearest.splice(at, 0, { shot, distance });
    if (nearest.length > count) nearest.pop();
  }
  return nearest.map((entry) => entry.shot);
}
