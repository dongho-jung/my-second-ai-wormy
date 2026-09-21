// One shape, two sources.
//
// An observation has to be buildable from the headless world an agent trains
// in and from the live game it will later play, or the policy meets different
// numbers on its first real match. So neither encoder reads the engine or the
// adapter directly: both read this, and the two builders below are the only
// places that know which side the numbers came from.
//
// The field names are the adapter's own, so `viewFromSnapshot` is a projection
// rather than a translation, and a mapping that changes upstream breaks in one
// obvious place.
import { terrainOf } from "./terrain.js";

/**
 * What a live room is assumed to be running at, in ticks.
 *
 * The room this project watches sits about 300 milliseconds behind, which at
 * sixty ticks a second is eighteen of them. A caller that has measured the
 * actual round trip should pass it instead.
 */
export const LIVE_LATENCY_TICKS = 18;

const vector = (x, y) => ({ x, y });

function decodeTerrainBytes(data) {
  if (typeof data === "string") return new Uint8Array(Buffer.from(data, "base64"));
  return data;
}

/**
 * The view of one worm in a headless world. `foes` is passed in rather than
 * read from `world.za`, because a worm that died this tick has already been
 * dropped from there and an agent still has to see that it is gone.
 */
export function viewFromWorld(world, self, foes = [], inputLatencyTicks = 0) {
  const level = world.level;
  return {
    tick: world.qb,
    // How late this worm's keys land. Drawn once per episode by the
    // environment and passed in, because the world does not know: the delay is
    // the connection's, not the game's.
    inputLatencyTicks,
    map: { name: level.name, width: level.width, height: level.height },
    terrain: terrainOf({
      data: level.data,
      width: level.width,
      height: level.height,
      materialFlags: world.s.Da,
    }),
    self: wormFromEngine(world, self),
    foes: foes.map((foe) => wormFromEngine(world, foe)),
    projectiles: [
      ...poolFromEngine(world.Ib, "weapon"),
      ...poolFromEngine(world.Zb, "particle", { ownedOnly: true }),
    ],
    // Health and weapon crates. A worm on 20 health with a medkit two ledges
    // away is in a different situation from one with none, and until now the
    // observation could not tell them apart.
    pickups: pickupsFromEngine(world),
  };
}

function pickupsFromEngine(world) {
  const out = [];
  const pool = world.Yb;
  for (let slot = 0; slot < pool.$; slot++) {
    const crate = pool.list[slot];
    if (!crate.u) continue;
    out.push({
      kind: crate.kind === 1 ? "health" : "weapon",
      weaponId: crate.kind === 1 ? null : crate.lf,
      position: vector(crate.x, crate.y),
    });
  }
  return out;
}

function wormFromEngine(world, worm) {
  if (!worm?.u) return { alive: false };
  return {
    alive: true,
    playerId: worm.H,
    position: vector(worm.x, worm.y),
    velocity: vector(worm.f, worm.b),
    health: worm.Xa,
    // Aim is stored as an angle off the facing direction; this is the same
    // screen-space angle the adapter reports, with y pointing down.
    aimRadians: worm.direction === 1 ? -worm.Oa : Math.PI + worm.Oa,
    facing: worm.direction === 1 ? "right" : "left",
    selectedWeapon: worm.Ka,
    weapons: worm.O.map((weapon, slot) => ({
      slot,
      id: weapon.type.id,
      name: weapon.type.name,
      ammo: weapon.ha,
      capacity: weapon.type.ha,
      reloadTicksRemaining:
        weapon.ha <= 0
          ? Math.max(0, Math.ceil(weapon.type.Vi * world.le - weapon.gb))
          : 0,
      cooldownTicksRemaining: Math.max(0, weapon.wd),
    })),
    rope: worm.Fa.Sc
      ? {
          position: vector(worm.Fa.x, worm.Fa.y),
          attached: worm.Fa.jc,
          length: worm.Fa.length,
        }
      : null,
  };
}

/**
 * The live objects a pool holds, as things a worm might have to care about.
 *
 * `ownedOnly` drops the ones no weapon threw. In this engine those are the
 * scenery: blood, spent shells, and the parts of a worm that has just come
 * apart. Over a whole episode not one of them could hurt anybody — every
 * particle with no weapon behind it had no hit damage, could not set off a
 * worm and left the ground alone — while the ones a weapon did throw, its
 * splinters and smoke, stay, because they mark where something just went off.
 *
 * Keeping the scenery was not free. Only the three nearest shots reach the
 * vector, and blood sprays from the worm that was just hit, so the scenery
 * took 83% of those slots and hid two thirds of the real incoming fire.
 */
function poolFromEngine(pool, kind, { ownedOnly = false } = {}) {
  const out = [];
  for (let slot = 0; slot < pool.$; slot++) {
    const entity = pool.list[slot];
    if (!entity.u) continue;
    if (ownedOnly && entity.La === 255) continue;
    out.push({
      kind,
      position: vector(entity.x, entity.y),
      velocity: vector(entity.f, entity.b),
      ownerPlayerId: entity.H < 0 ? null : entity.H,
      // Which weapon threw it, so the policy can tell a pellet from a rocket.
      weaponId: entity.La === 255 ? null : entity.La,
    });
  }
  return out;
}

/**
 * The same view from what the live game hands out: one `/state` sample and one
 * `/map` read. The terrain changes far more slowly than the state, so the two
 * are read on different clocks and passed in together here.
 */
export function viewFromSnapshot(
  state,
  terrain,
  // The live game has a real delay and no way to ask it what it is, so it is
  // told: eighteen ticks is the 300ms the watched room measures. Left at the
  // default, a policy plays as though the room behaved the way it usually does.
  { playerId = null, inputLatencyTicks = LIVE_LATENCY_TICKS } = {},
) {
  // Any player, not only the one at this keyboard: the game replicates
  // everybody's state, so a person's match can be watched and learned from
  // through a tab that is not theirs.
  const self =
    playerId === null
      ? state.players.find((player) => player.local)
      : state.players.find((player) => player.id === playerId);
  if (!self) throw new Error(`the snapshot has no player ${playerId ?? "(local)"}`);
  const worm = (player) =>
    player.worm
      ? { alive: true, playerId: player.id, ...player.worm }
      : { alive: false, playerId: player.id };
  return {
    tick: state.tick,
    inputLatencyTicks,
    map: state.map,
    terrain: terrainOf({
      data: decodeTerrainBytes(terrain.data),
      width: terrain.map.width,
      height: terrain.map.height,
      materialFlags: Uint8Array.from(terrain.materialFlags),
    }),
    self: worm(self),
    foes: state.players.filter((player) => player.id !== self.id).map(worm),
    // The same scenery the headless path drops, by the same rule: if no weapon
    // threw it, it cannot hurt anybody. A policy has to see the same world in
    // a live game as it did while learning.
    projectiles: state.projectiles
      .filter((shot) => shot.kind !== "particle" || shot.weaponId != null)
      .map((shot) => ({
        kind: shot.kind,
        position: shot.position,
        velocity: shot.velocity,
        ownerPlayerId: shot.ownerPlayerId,
        weaponId: shot.weaponId ?? null,
      })),
    pickups: (state.pickups ?? []).map((crate) => ({
      kind: crate.kind,
      weaponId: crate.weaponId,
      position: crate.position,
    })),
  };
}
