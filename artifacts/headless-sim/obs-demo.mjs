// What the two observation designs actually contain, printed from a real map.
import { readFileSync } from "node:fs";
import "./probe.mjs";
const wl = globalThis.__wl;
new Function(readFileSync("json5.min.js", "utf8") + ";globalThis.JSON5=JSON5;")();
const wasm = readFileSync("wasm-flate.wasm");
globalThis.fetch = async () => ({ arrayBuffer: async () => wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength) });
await wl.A.Pc("vendor/wasm-flate.wasm");
const zip = wl.ib.read(new Uint8Array(readFileSync("res.dat")));
const s = wl.U.dj(zip.get("mods/liero133/mod.json5").yl());
const spr = wl.Jc.read(new wl.H(new DataView(zip.get("mods/liero133/sprites.wlsprt").Ug()), true));
s.ba = spr.ba; s.Ha = spr.bj;
const settings = s.normalize();

const w = new wl.Pa(); w.s = settings;
const lev = readFileSync("Simple2.lev");
w.level.read("Simple2", lev.buffer.slice(lev.byteOffset, lev.byteOffset + lev.byteLength));
const lo = [0, 2, 3, 5, 10];
let me, foe;
// pick a spot that is actually interesting: standing on ground, mixed terrain
for (const seed of [1, 7, 12, 23, 44, 91, 130, 205, 377, 512, 808, 1234]) {
  w.za.length = 0; w.ca.x = seed;
  me = w.ox(0, 0, lo); foe = w.ox(1, 1, lo);
  for (let i = 0; i < 200; i++) { me.Wa = i % 60 < 30 ? 2 : 1; foe.Wa = 1; w.update(); }
  const L0 = w.level, Da0 = settings.Da;
  let free = 0, n = 0;
  for (let dy = -16; dy < 16; dy += 2) for (let dx = -32; dx < 32; dx += 2) {
    const x = (me.x | 0) + dx, y = (me.y | 0) + dy; n++;
    const fl = (x < 0 || y < 0 || x >= L0.width || y >= L0.height) ? 4 : Da0[L0.data[y * L0.width + x]];
    if (fl & 8) free++;
  }
  const grounded = ((Da0[L0.data[Math.min(L0.height - 1, (me.y | 0) + 4) * L0.width + (me.x | 0)]] & 8) === 0);
  if (grounded && free / n > 0.45 && free / n < 0.8) { console.log(`(seed ${seed})`); break; }
}

const L = w.level, Da = settings.Da;
const flagsAt = (x, y) => (x < 0 || y < 0 || x >= L.width || y >= L.height) ? 4 : Da[L.data[y * L.width + x]];
const solid = (x, y) => (flagsAt(x, y) & 8) === 0;        // bit 8 clear = solid
const rock = (x, y) => Boolean(flagsAt(x, y) & 4);        // stops shots

console.log(`worm at (${me.x | 0}, ${me.y | 0}) on ${L.name} ${L.width}x${L.height}\n`);
console.log("=== A. 지형 conv 가 보는 것: 웜 중심 64x32px, 2px 한 칸으로 줄여 표시 ===");
console.log("    (# 바위 = 총알도 막힘, : 흙 = 팔 수 있음, . 빈 공간, @ 나, X 상대)\n");
for (let dy = -16; dy < 16; dy += 2) {
  let row = "    ";
  for (let dx = -32; dx < 32; dx += 2) {
    const x = (me.x | 0) + dx, y = (me.y | 0) + dy;
    if (Math.abs(dx) <= 1 && Math.abs(dy) <= 2) row += "@";
    else if (Math.abs(x - (foe.x | 0)) <= 2 && Math.abs(y - (foe.y | 0)) <= 3) row += "X";
    else row += rock(x, y) ? "#" : solid(x, y) ? ":" : ".";
  }
  console.log(row);
}

console.log("\n=== B. 벡터 관측이 보는 것: 같은 지형을 16방향 거리 숫자로 ===\n");
const rays = [];
for (let k = 0; k < 16; k++) {
  const a = (k / 16) * Math.PI * 2, ux = Math.cos(a), uy = Math.sin(a);
  let d = 0;
  while (d < 120 && !solid((me.x + ux * d) | 0, (me.y + uy * d) | 0)) d += 2;
  rays.push(d);
}
console.log("    거리:", rays.map((d) => String(d).padStart(3)).join(" "));
console.log("    방향: ", ["→", "", "↘", "", "↓", "", "↙", "", "←", "", "↖", "", "↑", "", "↗", ""].map((x) => x.padStart(3)).join(" "));
const vec = {
  "나: 속도": [+me.f.toFixed(3), +me.b.toFixed(3)], "나: 체력": me.Xa, "나: 조준각": +me.Oa.toFixed(3),
  "나: 방향": me.direction, "나: 무기": settings.O[me.O[me.Ka].type.id].name, "나: 탄": me.O[me.Ka].ha,
  "상대: 상대좌표": [(foe.x - me.x) | 0, (foe.y - me.y) | 0], "상대: 체력": foe.Xa,
  "발밑 지형": solid(me.x | 0, (me.y | 0) + 4), "머리 위": solid(me.x | 0, (me.y | 0) - 4),
  "살아있는 투사체": w.Ib.$,
};
for (const [k, v] of Object.entries(vec)) console.log(`    ${k.padEnd(16)} ${JSON.stringify(v)}`);
console.log(`\n    -> conv 쪽 입력 크기: 3채널 x 32 x 32 = ${3 * 32 * 32} 숫자`);
console.log(`    -> 벡터 쪽 입력 크기: 레이 16 + 상태 ~40 = 56 숫자`);
