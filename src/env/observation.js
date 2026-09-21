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
import { KEYS } from "./actions.js";
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
export const WEAPON_SLOTS = 5;
/**
 * What a delay is measured against, in ticks.
 *
 * The room this plays in runs about 300ms behind — eighteen ticks — and
 * training draws from a range around that. Dividing by a fixed number rather
 * than by the range actually in use keeps the field meaning the same thing
 * across runs, so a policy trained at 6-21 and played at a measured 24 reads
 * "later than anything I saw" instead of "the usual".
 */
export const LATENCY_SCALE_TICKS = 30;
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
 * The previous decision, as the engine sees it: the nine held keys, in the
 * order `KEYS` lists them, then the rope message (-1 release, 1 throw) and the
 * weapon message (-1 previous, 1 next).
 */
const LAST_ACTION_KEYS = Object.values(KEYS);
export const LAST_ACTION_FIELDS = LAST_ACTION_KEYS.length + 2;

/**
 * The vector, field by field. Exported because a layout you cannot print is a
 * layout you cannot debug: `VECTOR_OFFSETS.foe` is where the other worm starts.
 */
/** Pixels per patch cell unless a run says otherwise. */
const DEFAULT_PATCH_SCALE = 2;
/** What a patch is made of, one plane per kind. */
const PATCH_CHANNELS = ["rock", "dirt", "free", "projectile", "foe", "self"];
/**
 * How much ground a patch shows, in pixels: 426 wide and 242 high around the
 * worm, which on a 504 x 350 map is most of it. Every scale sees this much;
 * what changes is how many cells it takes to say so.
 */
const PATCH_PX = { width: 426, height: 242 };

/**
 * The patch's cells for a given number of pixels per cell.
 *
 * At 2 it is the 213 x 121 this project has always used, the resolution
 * footwork happens at. At 4 it is 107 x 61, the same picture in a quarter of
 * the cells — and the convolution over those cells is most of what an update
 * costs, so this is the dial for trading detail against samples per hour.
 * A cell still answers for the hardest of its pixels, so a wall one pixel
 * thick does not vanish at any scale.
 */
export function patchGeometry(scalePx = DEFAULT_PATCH_SCALE) {
  if (!Number.isInteger(scalePx) || scalePx < 1) {
    throw new Error(`patch scale must be a whole number of pixels, got ${scalePx}`);
  }
  const columns = Math.ceil(PATCH_PX.width / scalePx);
  const rows = Math.ceil(PATCH_PX.height / scalePx);
  const cells = columns * rows;
  return {
    channels: PATCH_CHANNELS,
    columns,
    rows,
    scalePx,
    cells,
    size: PATCH_CHANNELS.length * cells,
    /** The shape a convolution reads it as: channels, rows, columns. */
    shape: [PATCH_CHANNELS.length, rows, columns],
  };
}

  // Free space gets a channel of its own instead of being the absence of the
  // other two, so "solid" and "nothing measured" never look alike. Past the
  // edge of the map reads as rock, which is exactly how it behaves.
  // Worms among the terrain, not only as coordinates in the vector. Without
  // these the policy sees the ground in front of it and is told where the enemy
  // is in two numbers it has to reconcile with that picture itself; "he is
  // behind that wall" is something it should be able to look at.
export const PATCH = patchGeometry(DEFAULT_PATCH_SCALE);
export const PATCH_CELLS = PATCH.cells;
export const PATCH_SIZE = PATCH.size;
export const PATCH_SHAPE = PATCH.shape;

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
  patchScale = DEFAULT_PATCH_SCALE,
  // A second cut of the same ground, for a match in which two policies were
  // trained at different scales: each is shown the patch it learned on.
  patchScale2 = null,
} = {}) {
  const layout = [
    ["rays", RAY_COUNT], //   distance to the first solid pixel, 1 = clear to the limit
    ["health", 1],
    ["velocity", 2], //       px per tick, already around 1
    ["aim", 2], //            cos and sin, so the wrap at PI is not a cliff
    // How fast the aim is turning, radians per tick. The aim keys accelerate
    // the angle rather than moving it a fixed step, so without this a policy
    // sees where it is pointing and not where it is about to point, and
    // overshoots what it cannot see coming.
    ["aimVelocity", 1],
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
    // What it chose last decision. Nothing in the world says whether jump or
    // dig was already held — both fire on the press and go dead while held —
    // or what is still in the input-delay queue on its way to the worm, and a
    // memory built only from observations cannot recover a choice it never
    // saw. Written as the engine's own keys and messages, so a person's
    // recorded play fills it the same way.
    ["lastAction", LAST_ACTION_FIELDS],
    // How far behind this worm is playing, as a share of the longest delay the
    // environment will hand out. A worm that has to lead its shots by twenty
    // ticks is playing a different game from one that acts immediately, and
    // until now nothing said which it was: the delay is drawn once per episode
    // and then never mentioned, so the policy had to infer it from how late
    // the world kept reacting.
    ["latency", 1],
    // Which weapon, not just what it does. Twelve names in this mod belong to
    // two different weapons — the ordinary one and the strange crate-only
    // version — and ten measured numbers do not say how a VIRUS spreads or
    // what a FORCE FIELD is for. These are indices for the network to embed,
    // so they are kept out of the running normaliser, which would average them
    // into nonsense.
    //
    // All five of my slots, then each foe's held one. Only the held weapon used
    // to be in here, which left the policy choosing "next" and "previous"
    // through five slots it could not see the contents of — it knew a slot had
    // four rounds and was ready to fire, and not whether it was a shotgun or a
    // bazooka. Spawning and picking up a crate both change slots with nothing
    // to notice it by, so no amount of memory recovers it. -1 for an empty slot.
    ["weaponIds", WEAPON_SLOTS + foeSlots],
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
    patch: patchGeometry(patchScale),
    patch2: patchScale2 ? patchGeometry(patchScale2) : null,
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
  if (kinds.includes("patch")) out.patch = encodePatch(view, into.patch, spec.patch);
  if (kinds.includes("patchBytes")) {
    out.patchBytes = encodePatchBytes(view, into.patchBytes, spec.patch);
  }
  if (kinds.includes("patchBytes2") && spec.patch2) {
    out.patchBytes2 = encodePatchBytes(view, into.patchBytes2, spec.patch2);
  }
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
  into[at++] = self.aimVelocity ?? 0;
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
    const from = weaponFeaturesAt(spec, shot.weaponId);
    for (const index of [4, 6, 1, 0]) {
      into[at++] = from >= 0 ? features[from + index] : 0;
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

  const last = view.lastAction;
  if (last) {
    for (const bit of LAST_ACTION_KEYS) into[at++] = last.keys & bit ? 1 : 0;
    into[at++] = last.rope ?? 0;
    into[at++] = last.weapon ?? 0;
  } else {
    at += LAST_ACTION_FIELDS;
  }

  // Which weapon, as an index for the network to embed rather than a number to
  // do arithmetic on. Written last and kept out of the running normaliser,
  // which would otherwise average weapon 31 and weapon 108 into weapon 70 —
  // and those two are a VIRUS that spreads a little and one that spreads twice
  // as far. Ten measured numbers cannot say that; an identity can.
  // How late this worm's keys arrive, against the longest delay there is.
  into[at++] = clamp((view.inputLatencyTicks ?? 0) / LATENCY_SCALE_TICKS, 0, 1);

  const held = (worm) => worm?.weapons?.[worm.selectedWeapon]?.id ?? -1;
  for (let slot = 0; slot < WEAPON_SLOTS; slot++) {
    into[at++] = self.weapons?.[slot]?.id ?? -1;
  }
  for (let slot = 0; slot < spec.foeSlots; slot++) into[at++] = held(foes[slot]);

  return into;
}

/**
 * Where one weapon's measured character starts in the table, or -1 for none.
 *
 * Reading past the end of a Float32Array gives `undefined`, and writing that
 * into another one gives NaN rather than throwing. A live room running a mod
 * whose weapon ids outrun the measured table therefore produced observations
 * that looked fine, were recorded, and turned the whole policy to NaN on the
 * first update that learned from them. Out of range is zeros, like no weapon.
 */
function weaponFeaturesAt(spec, weaponId) {
  const features = spec.weaponFeatures;
  if (!features || weaponId == null || weaponId < 0) return -1;
  const from = weaponId * WEAPON_FEATURE_COUNT;
  return from + WEAPON_FEATURE_COUNT <= features.length ? from : -1;
}

/** One weapon's measured character, or zeros when there is nothing to say. */
function writeWeapon(into, at, spec, weaponId) {
  const features = spec.weaponFeatures;
  const from = weaponFeaturesAt(spec, weaponId);
  for (let index = 0; index < WEAPON_FEATURE_COUNT; index++) {
    into[at + index] = from >= 0 ? features[from + index] : 0;
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
export function patchOriginOf(view, geometry = PATCH) {
  const { columns, rows, scalePx } = geometry;
  const half = scalePx >> 1;
  return {
    x: Math.round(view.self.position.x) - ((columns >> 1) * scalePx + half),
    y: Math.round(view.self.position.y) - ((rows >> 1) * scalePx + half),
  };
}

/** Which cell a pixel falls in, or `null` when it is outside the patch. */
export function patchCellOf(origin, x, y, geometry = PATCH) {
  const column = Math.floor((x - origin.x) / geometry.scalePx);
  const row = Math.floor((y - origin.y) / geometry.scalePx);
  if (column < 0 || row < 0 || column >= geometry.columns || row >= geometry.rows)
    return null;
  return { column, row, cell: row * geometry.columns + column };
}

// Scratch, reused between calls. The level row each of the patch's pixel rows
// lands on, and the level x of each of its pixel columns, with -1 for the ones
// that fall off the map.
// Scratch for the encoder, one set per geometry in use: the pixel-to-row and
// pixel-to-column tables, and a byte patch for the one-hot form to expand from.
const scratchFor = new Map();
function scratch(geometry) {
  let found = scratchFor.get(geometry.scalePx);
  if (!found) {
    found = {
      pixelRows: new Int32Array(geometry.rows * geometry.scalePx),
      pixelColumns: new Int32Array(geometry.columns * geometry.scalePx),
      bytes: new Uint8Array(geometry.cells),
    };
    scratchFor.set(geometry.scalePx, found);
  }
  return found;
}
// The first three channels are the hardness order, so the lowest code wins and
// is also the channel it expands to.
const ROCK = 0;
const DIRT = 1;
const FREE = 2;
/** Bit set on a cell that has a shot in it, alongside the terrain code. */
export const PATCH_PROJECTILE = 4;
export const PATCH_FOE = 8;
export const PATCH_SELF = 16;
export const PATCH_KIND = 3;

/**
 * The patch as one byte per cell — the form it is stored and sent in.
 *
 * Bits 0-1 are the terrain (0 rock, 1 dirt, 2 free) and bit 2 says a shot is in
 * the cell. A quarter of the size of the one-hot form and the same information:
 * a trainer expands it on the GPU, where the expansion is free, and a run that
 * ships a million of these across a pipe ships a quarter of the bytes.
 */
export function encodePatchBytes(view, into, geometry = PATCH) {
  into ??= new Uint8Array(geometry.cells);
  into.fill(0);
  const { self, terrain } = view;
  if (!self.alive) return into;
  const { columns, rows, scalePx } = geometry;
  const { pixelRows: PIXEL_ROWS, pixelColumns: PIXEL_COLUMNS } = scratch(geometry);
  const { data, width, height, materialFlags } = terrain;
  const origin = patchOriginOf(view, geometry);
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
    const at = patchCellOf(origin, shot.position.x, shot.position.y, geometry);
    if (at) into[at.cell] |= PATCH_PROJECTILE;
  }
  // And the worms, in the same picture as the ground they stand on. The vector
  // says where a foe is in two numbers; this says it in the place it actually
  // is, beside the wall that is or is not between them.
  for (const foe of view.foes ?? []) {
    if (!foe.alive) continue;
    const at = patchCellOf(origin, foe.position.x, foe.position.y, geometry);
    if (at) into[at.cell] |= PATCH_FOE;
  }
  const here = patchCellOf(origin, self.position.x, self.position.y, geometry);
  if (here) into[here.cell] |= PATCH_SELF;
  return into;
}

/** The same patch as one-hot float planes, for a consumer that wants them. */
export function encodePatch(view, into, geometry = PATCH) {
  into ??= new Float32Array(geometry.size);
  into.fill(0);
  if (!view.self.alive) return into;
  const bytes = encodePatchBytes(view, scratch(geometry).bytes, geometry);
  const plane = geometry.cells;
  for (let cell = 0; cell < plane; cell++) {
    const byte = bytes[cell];
    into[(byte & PATCH_KIND) * plane + cell] = 1;
    if (byte & PATCH_PROJECTILE) into[3 * plane + cell] = 1;
    if (byte & PATCH_FOE) into[4 * plane + cell] = 1;
    if (byte & PATCH_SELF) into[5 * plane + cell] = 1;
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
