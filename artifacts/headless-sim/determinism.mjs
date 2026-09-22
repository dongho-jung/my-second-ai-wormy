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
const episode = (seed) => {
  const w = new wl.Pa(); w.s = settings;
  const lev = readFileSync("Simple2.lev");
  w.level.read("Simple2", lev.buffer.slice(lev.byteOffset, lev.byteOffset + lev.byteLength));
  w.ca.x = seed >>> 0;
  const lo = [0, 2, 3, 5, 10];
  const worms = [w.ox(0, 0, lo), w.ox(1, 1, lo)];
  let rng = 7;
  const rand = () => (rng = (1664525 * rng + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < 20000; i++) {
    for (const worm of worms) { if (i % 4 === 0) worm.__a = (rand() < .5 ? 1 : 2) | (rand() < .4 ? 4 : 0) | (rand() < .3 ? 16 : 0); worm.Wa = worm.u ? worm.__a : 0; }
    w.update();
  }
  return JSON.stringify(worms.map((x) => [+x.x.toFixed(6), +x.y.toFixed(6), +x.Xa.toFixed(4), x.u]));
};
const a = episode(99), b = episode(99), c = episode(100);
console.log("same seed twice identical:", a === b, "| different seed differs:", a !== c);
console.log("state:", a);
