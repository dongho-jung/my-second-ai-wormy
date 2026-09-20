// Download the level pool the game itself offers, and keep the ones worth
// training on.
//
// The bundle carries the whole catalogue as names and category flags, and the
// site serves the files. "Best" is the curated 52 a room's host is most likely
// to pick, which is exactly the distribution a policy has to survive.
//
// A few of them are unplayable — the engine picks a spawn by looking for dirt
// or background and gives up after a hundred tries, so a level that is nearly
// all rock leaves worms stacked in a corner with nothing to learn from. Those
// are measured here and left out, with the reason printed.
import { parseArgs } from "node:util";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { loadEngine, ASSETS, DEFAULT_ENGINE_DIR } from "../src/env/engine.js";

const HELP = `Wormy II — fetch the game's own level pool

Usage: npm run levels -- [options]

  --category best   best (the curated 52) or all (every level listed)
  --out PATH        where to put them (default artifacts/levels)
  --min-open 0.08   least share of the map that has to be somewhere a worm can
                    spawn; below that the engine cannot place them
  --keep-broken     download the unplayable ones too, without using them
  --pause 400       milliseconds between requests; the service rate-limits`;

const { values } = parseArgs({
  options: {
    help: { type: "boolean", short: "h" },
    category: { type: "string", default: "best" },
    out: { type: "string" },
    "min-open": { type: "string", default: "0.08" },
    "keep-broken": { type: "boolean", default: false },
    pause: { type: "string", default: "400" },
  },
  allowNegative: true,
});
if (values.help) {
  console.log(HELP);
  process.exit(0);
}

// The categories the game's own list uses. `Oc(mask)` keeps a level when its
// flags share a bit with the mask, and keeps everything when the mask is zero.
const CATEGORIES = { best: 256, all: 0 };
const mask = CATEGORIES[values.category];
if (mask === undefined) {
  throw new Error(`--category must be one of ${Object.keys(CATEGORIES).join(", ")}`);
}
const minOpen = Number(values["min-open"]);
const out = new URL(
  values.out ? `${values.out}/` : "../artifacts/levels/",
  values.out ? `file://${process.cwd()}/` : import.meta.url,
);
const BASE = "https://api.webliero.com/__cache_static__/levels/";
const pause = Number(values.pause);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * One level file. The service rate-limits, and the honest answer to a 429 is to
 * wait rather than to hammer it: the whole pool is fifty-odd files fetched once.
 */
async function download(name, attempt = 0) {
  const response = await fetch(BASE + encodeURIComponent(name));
  if (response.status === 429 && attempt < 6) {
    const backoff = pause * 2 ** (attempt + 1);
    await wait(backoff);
    return download(name, attempt + 1);
  }
  if (!response.ok) return { error: `download failed (${response.status})` };
  return { bytes: Buffer.from(await response.arrayBuffer()) };
}

/** The catalogue, read out of the bundle rather than guessed at. */
async function catalogue() {
  const source = await readFile(new URL(ASSETS.bundle, DEFAULT_ENGINE_DIR), "utf8");
  const at = source.indexOf("F.$l=[");
  if (at < 0) throw new Error("the bundle has no level list: is this v20?");
  const list = new Function(`return ${source.slice(at + 5, source.indexOf("];", at) + 1)}`)();
  const levels = [];
  for (let index = 0; index < list.length; index += 2) {
    levels.push({ name: list[index], flags: list[index + 1] });
  }
  return levels.filter((level) => mask === 0 || (level.flags & mask) !== 0);
}

const engine = await loadEngine();
const wanted = await catalogue();
await mkdir(out, { recursive: true });
console.log(`${wanted.length} levels in "${values.category}"`);

const flags = engine.materialFlags;
// The engine looks for a pixel that is dirt or background when it places a
// worm. Rock is neither, so a level made of it has nowhere to spawn.
const SPAWNABLE = 11;

const kept = [];
const skipped = [];
for (const [index, level] of wanted.entries()) {
  const file = new URL(encodeURIComponent(level.name), out);
  let bytes;
  try {
    bytes = await readFile(file);
  } catch {
    const got = await download(level.name);
    if (got.error) {
      skipped.push({ name: level.name, why: got.error });
      continue;
    }
    bytes = got.bytes;
    await writeFile(file, bytes);
    await wait(pause);
  }
  let open = 0;
  let air = 0;
  try {
    const read = engine.readLevel(level.name.replace(/\.lev$/i, ""), bytes);
    for (const index of read.data) {
      const material = flags[index];
      if (material & SPAWNABLE) open++;
      if (material & 8) air++;
    }
    const share = open / read.data.length;
    if (share < minOpen) {
      skipped.push({ name: level.name, why: `only ${(share * 100).toFixed(1)}% of it can hold a worm` });
      continue;
    }
    kept.push({ name: level.name, open: share, air: air / read.data.length });
  } catch (error) {
    skipped.push({ name: level.name, why: error.message });
  }
  if ((index + 1) % 10 === 0) console.log(`  ${index + 1}/${wanted.length}`);
}

console.log(`\nkept ${kept.length}:`);
for (const level of kept) {
  console.log(
    `  ${level.name.padEnd(28)} ${(level.open * 100).toFixed(0).padStart(3)}% spawnable, ${(level.air * 100).toFixed(0).padStart(3)}% open air`,
  );
}
if (skipped.length) {
  console.log(`\nleft out ${skipped.length}:`);
  for (const level of skipped) console.log(`  ${level.name.padEnd(28)} ${level.why}`);
}
await writeFile(
  new URL("levels.json", out),
  `${JSON.stringify({ category: values.category, kept, skipped }, null, 2)}\n`,
);
console.log(`\n-> ${out.pathname}`);
