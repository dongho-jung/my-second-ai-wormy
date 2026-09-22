// Boot the shipped WebLiero v20 engine in plain Node — no browser, no server,
// no rendering — and run its own world.update() as fast as the CPU allows.
import { readFileSync } from "node:fs";
import "./probe.mjs";                       // DOM stubs + engine.js
const wl = globalThis.__wl;

// json5 is a page vendor script the mod loader calls through the global.
new Function(readFileSync("json5.min.js", "utf8") + ";globalThis.JSON5=JSON5;")();

// The engine inflates res.dat through wasm-flate, loaded with fetch().
const wasmBytes = readFileSync("wasm-flate.wasm");
globalThis.fetch = async () => ({ arrayBuffer: async () => wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) });
await wl.A.Pc("vendor/wasm-flate.wasm");
console.log("wasm loaded:", typeof wl.A.Sb?.zlib_decode_raw);

// res.dat is a zip of mods; liero133 is the stock Liero 1.33 rule set.
const res = readFileSync("res.dat");
const zip = wl.ib.read(new Uint8Array(res));
console.log("res.dat entries:", [...zip.keys()].length);
const mod = (name) => {
  const s = wl.U.dj(zip.get(`mods/${name}/mod.json5`).yl());
  const spr = zip.get(`mods/${name}/sprites.wlsprt`).Ug();
  const read = wl.Jc.read(new wl.H(new DataView(spr), true));
  s.ba = read.ba; s.Ha = read.bj;
  return s.normalize();
};
const settings = mod("liero133");
console.log("mod:", settings.name, "weapons:", settings.O.length, "gravity:", settings.pb.wormGravity ?? Object.keys(settings.pb).length);

// A world, a real Liero map, two worms.
const world = new wl.Pa();
world.s = settings;
const lev = readFileSync("Simple2.lev");
world.level.read("Simple2", lev.buffer.slice(lev.byteOffset, lev.byteOffset + lev.byteLength));
world.ca.x = 12345;                          // seeded LCG: deterministic
console.log("level:", world.level.name, world.level.width + "x" + world.level.height);
const loadout = [0, 1, 2, 3, 4];
const a = world.ox(0, 0, loadout);
const b = world.ox(1, 1, loadout);
console.log("loadout:", a.O.map((s) => s.type.name).join(", "));
console.log("worms:", world.za.length, JSON.stringify({ a: { x: a.x, y: a.y, hp: a.Xa }, b: { x: b.x, y: b.y, hp: b.Xa } }));

// Inputs are one bitmask per worm (worm.Wa), read by update().
const step = (n, bitsA = 0, bitsB = 0) => { for (let i = 0; i < n; i++) { a.Wa = bitsA; b.Wa = bitsB; world.update(); } };
step(60);
console.log("after 60 idle ticks:", JSON.stringify({ a: { x: +a.x.toFixed(2), y: +a.y.toFixed(2), hp: a.Xa }, tick: world.qb }));
const before = { x: a.x, y: a.y };
step(120, 2, 1);                             // walk right / walk left
console.log("after 120 walking ticks: dx =", +(a.x - before.x).toFixed(2), "b dx =", +(b.x).toFixed(2));

// Throughput, with shooting so projectiles and terrain damage are in the loop.
for (const [label, bitsA, bitsB] of [["idle", 0, 0], ["walk", 2, 1], ["walk+fire", 2 | 16, 1 | 16]]) {
  const t0 = performance.now();
  const N = 200000;
  step(N, bitsA, bitsB);
  const ms = performance.now() - t0;
  console.log(`${label}: ${N} ticks in ${Math.round(ms)} ms = ${Math.round(N / (ms / 1000)).toLocaleString()} ticks/s  (${(N / (ms / 1000) / 60).toFixed(0)}x realtime)`);
}
console.log("final:", JSON.stringify({ tick: world.qb, aHp: a.Xa, bHp: b.Xa, projectiles: world.Ib.$, alive: world.za.map((w) => w.u) }));
