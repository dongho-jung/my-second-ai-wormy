// What each of the mod's weapons actually does, measured by firing it.
//
// A policy that sees only "slot 3, four rounds left" has to learn what slot 3
// is from scratch on every map, and cannot tell a rifle from a mine until it
// has blown itself up with one. The mod knows the answer, but it knows it in
// minified fields whose meanings are guesses. So this fires every weapon in a
// controlled world and measures what happens: how fast the shot goes, whether
// it arcs, how far it gets, what it does to a target, and what it does to the
// worm that fired it.
//
// The result is cached to artifacts/weapons.json and folded into the
// observation, so a worm is told what it is holding and what it is being shot
// with.
import { writeFile } from "node:fs/promises";
import { loadEngine, makeRng } from "../src/env/engine.js";
import { KEYS } from "../src/env/actions.js";

const TICKS = 300;
const engine = await loadEngine();
const settings = engine.settings;
const flags = engine.materialFlags;

/** A world with a clear corridor, so a shot is measured and not absorbed. */
function arena() {
  const level = engine.randomLevel(4242);
  // Hollow out a wide band: every measurement wants the same empty room.
  const { data, width, height } = level;
  // Open air has to be open to a shot as well as to a worm, and those are
  // different bits: bit 3 says a worm may stand there, bits 0-2 are what a
  // projectile collides with. An index with both set is background a worm walks
  // through and a rocket detonates against — which is what made every weapon
  // here look like it explodes in its own muzzle.
  const background = (() => {
    for (let index = 0; index < 256; index++) {
      if ((flags[index] & 8) !== 0 && (flags[index] & 7) === 0) return index;
    }
    throw new Error("the mod has no material that is open to both a worm and a shot");
  })();
  for (let y = 40; y < height - 40; y++) {
    for (let x = 10; x < width - 10; x++) data[y * width + x] = background;
  }
  const world = engine.createWorld({ level, seed: 7, rules: { bonusDrops: 0 } });
  return world;
}

const loadout = (id) => [id, id, id, id, id];

/** Fire one weapon down an empty room and watch where the shot goes. */
function flightOf(id) {
  const world = arena();
  const worm = engine.spawnWorm(world, { color: 0, playerId: 0, loadout: loadout(id) });
  worm.x = 60;
  worm.y = 175;
  worm.f = 0;
  worm.b = 0;
  worm.direction = 1;
  worm.Oa = 0; // straight ahead
  worm.Ka = 0;
  const start = { x: worm.x, y: worm.y };
  let live = 0;
  let farthest = 0;
  let drop = 0;
  let firstSpeed = 0;
  let alive = 0;
  let shots = 0;
  for (let tick = 0; tick < TICKS; tick++) {
    // Hold the worm still: this measures the shot, not the recoil.
    worm.x = start.x;
    worm.y = start.y;
    worm.f = 0;
    worm.b = 0;
    worm.Oa = 0;
    worm.direction = 1;
    worm.Wa = tick === 0 ? KEYS.fire : 0;
    world.update();
    let count = 0;
    for (const pool of [world.Ib, world.Zb]) {
      for (let slot = 0; slot < pool.$; slot++) {
        const shot = pool.list[slot];
        if (!shot.u || shot.H !== 0) continue;
        count++;
        const reach = shot.x - start.x;
        if (reach > farthest) farthest = reach;
        // How far it has fallen from the line it was fired along.
        const fall = shot.y - start.y;
        if (fall > drop) drop = fall;
        if (tick === 0) firstSpeed = Math.max(firstSpeed, Math.hypot(shot.f, shot.b));
      }
    }
    if (tick === 0) shots = count;
    if (count > 0) alive = tick + 1;
    live = Math.max(live, count);
  }
  return { shots, live, farthest, drop, firstSpeed, lifetime: alive };
}

/** What it does to somebody else, and to the worm holding it. */
function damageOf(id, distance) {
  const world = arena();
  const attacker = engine.spawnWorm(world, { color: 0, playerId: 0, loadout: loadout(id) });
  const target = engine.spawnWorm(world, { color: 1, playerId: 1, loadout: loadout(id) });
  const place = () => {
    attacker.x = 60;
    attacker.y = 175;
    attacker.f = attacker.b = 0;
    attacker.direction = 1;
    attacker.Oa = 0;
    attacker.Ka = 0;
    target.x = 60 + distance;
    target.y = 175;
    target.f = target.b = 0;
    target.Wa = 0;
  };
  place();
  for (let tick = 0; tick < TICKS; tick++) {
    place();
    attacker.Wa = tick === 0 ? KEYS.fire : 0;
    world.update();
    if (!attacker.u || !target.u) break;
  }
  return {
    dealt: 100 - (target.u ? target.Xa : 0),
    self: 100 - (attacker.u ? attacker.Xa : 0),
  };
}

const profiles = [];
for (let id = 0; id < settings.O.length; id++) {
  const type = settings.O[id];
  const flight = flightOf(id);
  const near = damageOf(id, 24);
  const far = damageOf(id, 140);
  profiles.push({
    id,
    name: type.name,
    capacity: type.ha,
    reloadTicks: Math.round(type.Vi * 0.4),
    fireDelay: type.delay,
    shots: flight.shots,
    speed: Number(flight.firstSpeed.toFixed(3)),
    // A shot that ends far below the line it was fired along is a thrown one.
    dropPx: Math.round(flight.drop),
    rangePx: Math.round(flight.farthest),
    lifetimeTicks: flight.lifetime,
    damageNear: Number(near.dealt.toFixed(1)),
    damageFar: Number(far.dealt.toFixed(1)),
    selfNear: Number(near.self.toFixed(1)),
  });
  process.stdout.write(`\r  ${id + 1}/${settings.O.length}`);
}
process.stdout.write("\r");

const out = new URL("../artifacts/weapons.json", import.meta.url);
await writeFile(out, `${JSON.stringify({ mod: settings.name, weapons: profiles }, null, 2)}\n`);

const show = (w) =>
  `${String(w.id).padStart(2)} ${w.name.padEnd(15)} spd ${w.speed.toFixed(1).padStart(5)} ` +
  `range ${String(w.rangePx).padStart(4)} drop ${String(w.dropPx).padStart(4)} ` +
  `shots ${String(w.shots).padStart(2)} | dmg near ${String(w.damageNear).padStart(5)} far ${String(w.damageFar).padStart(5)} ` +
  `| self ${String(w.selfNear).padStart(5)}`;
for (const weapon of profiles) console.log(show(weapon));
console.log(`\n-> ${out.pathname}`);
