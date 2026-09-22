// An honest throughput number: two worms that stay alive, move, aim and shoot,
// with respawns — the loop an RL environment would actually run.
import { readFileSync } from "node:fs";
import "./probe.mjs";
const wl = globalThis.__wl;
new Function(readFileSync("json5.min.js", "utf8") + ";globalThis.JSON5=JSON5;")();
const wasmBytes = readFileSync("wasm-flate.wasm");
globalThis.fetch = async () => ({ arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) });
await wl.A.Pc("vendor/wasm-flate.wasm");
const zip = wl.ib.read(new Uint8Array(readFileSync("res.dat")));
const s = wl.U.dj(zip.get("mods/liero133/mod.json5").yl());
const spr = wl.Jc.read(new wl.H(new DataView(zip.get("mods/liero133/sprites.wlsprt").Ug()), true));
s.ba = spr.ba; s.Ha = spr.bj;
const settings = s.normalize();

const makeWorld = (levelFile, seed) => {
  const w = new wl.Pa();
  w.s = settings;
  const lev = readFileSync(levelFile);
  w.level.read(levelFile, lev.buffer.slice(lev.byteOffset, lev.byteOffset + lev.byteLength));
  w.ca.x = seed >>> 0;
  const loadout = [0, 2, 3, 5, 10];
  const worms = [w.ox(0, 0, loadout), w.ox(1, 1, loadout)];
  return { w, worms, loadout };
};

// LEFT 1, RIGHT 2, AIM_UP 4, AIM_DOWN 8, FIRE 16, JUMP 32 (read off worm.update)
const run = (ticks, label, levelFile = "Simple2.lev") => {
  const { w, worms, loadout } = makeWorld(levelFile, 99);
  let rng = 1234567, deaths = 0, shots = 0;
  const rand = () => (rng = (1664525 * rng + 1013904223) >>> 0) / 4294967296;
  const t0 = performance.now();
  for (let i = 0; i < ticks; i++) {
    for (const worm of worms) {
      if (!worm.u) { worm.u = true; worm.nx(w, loadout); deaths++; continue; }
      // change the action every 4 ticks, the way a policy at 15 Hz would
      if (i % 4 === 0) worm.__a = (rand() < 0.5 ? 1 : 2) | (rand() < 0.4 ? 4 : 0) | (rand() < 0.25 ? 16 : 0) | (rand() < 0.1 ? 32 : 0);
      worm.Wa = worm.__a ?? 0;
      if (worm.Wa & 16) shots++;
    }
    w.update();
  }
  const ms = performance.now() - t0;
  const tps = ticks / (ms / 1000);
  console.log(`${label}: ${ticks.toLocaleString()} ticks in ${Math.round(ms)} ms = ${Math.round(tps).toLocaleString()} ticks/s = ${(tps / 60).toFixed(0)}x realtime | respawns ${deaths}, live projectiles ${w.Ib.$}, hp ${worms.map((x) => Math.round(x.Xa)).join("/")}`);
  return tps;
};

run(50000, "warmup");
const rates = [run(300000, "fight  Simple2"), run(300000, "fight  Arena_1", "Arena_1.lev"), run(300000, "fight  lb_infst", "lb_infst.lev")];
const avg = rates.reduce((a, b) => a + b, 0) / rates.length;
console.log(`\n1 core: ${Math.round(avg).toLocaleString()} ticks/s -> at 4-tick frameskip ${Math.round(avg / 4).toLocaleString()} agent steps/s`);
console.log(`hours of game time per wall-clock hour, 1 core: ${(avg / 60 / 60 / 60 * 3600 / 3600 * 3600 / 3600).toFixed(0)}`);
