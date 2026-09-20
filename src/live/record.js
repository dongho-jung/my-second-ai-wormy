// Watching people play, so the policy can be shown rather than only paid.
//
// Random exploration does not find the rope. Throwing it, holding a direction,
// shortening at the right moment and letting go is a sequence that has to be
// almost right before any of it pays, and a policy that has never seen it done
// has no reason to try. A few minutes of somebody playing skips past that.
//
// Nothing has to be installed on the player's side and they do not have to do
// anything except play: the game replicates every player's input to everyone in
// the room, so what they pressed arrives in the same field the policy writes.
// This reads it from a tab that is already there.
//
// Records are the same observation the trainer uses, byte for byte, so a
// demonstration and a rollout are the same kind of thing.
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { chromium } from "playwright";
import { WebLieroObserver } from "../observer.js";
import { createLogger } from "../log.js";
import { joinRoom, spawned } from "../room.js";
import { ACTION_HEADS } from "../env/actions.js";
import { DEFAULT_MOD, loadEngine } from "../env/engine.js";
import {
  MAP_SIZE,
  PATCH_CELLS,
  PATCH_SHAPE,
  encodeMap,
  encodeMapTerrain,
  observationSpec,
  observe,
} from "../env/observation.js";
import { viewFromSnapshot } from "../env/view.js";
import { headsFromKeys } from "./keys.js";

const HELP = `Wormy II — record people playing

Usage: npm run record -- [options]

Watches a room through a tab that is already in it and writes down what every
player with a living worm saw and pressed. Join the room and play; that is all
it needs. Stop it with Ctrl+C.

  --cdp-port 9334     the Chromium the game is in
  --hz 15             samples a second; 15 is the rate the policy decides at
  --map-ms 2000       how often the whole level is re-read
  --observation-foes  size the vector for this many foes (default 2)
  --out PATH          where the demonstration goes (default artifacts/demos)
  --exclude NAMES     comma-separated player names to ignore, e.g. the driven ones
  --idle-seconds 10   after this long with nobody playing, close the recording
                      and start learning from it
  --no-learn          record only; do not start cloning when play stops`;

const { values } = parseArgs({
  options: {
    help: { type: "boolean", short: "h" },
    "cdp-port": { type: "string", default: "9334" },
    hz: { type: "string", default: "15" },
    "map-ms": { type: "string", default: "2000" },
    "observation-foes": { type: "string", default: "2" },
    out: { type: "string" },
    exclude: { type: "string", default: "" },
    "idle-seconds": { type: "string", default: "10" },
    learn: { type: "boolean", default: true },
    "min-samples": { type: "string", default: "600" },
    mod: { type: "string", default: DEFAULT_MOD },
    "room-url": { type: "string" },
    "min-players": { type: "string", default: "2" },
    "idle-run": { type: "string", default: "45" },
    nickname: { type: "string", default: "OBSERVER" },
  },
  allowNegative: true,
});
if (values.help) {
  console.log(HELP);
  process.exit(0);
}

const log = createLogger({ scope: "record" });
const excluded = new Set(
  [
    // Whatever this tab is called. It is here to watch, and a recording of the
    // watcher is a recording of somebody sitting still in the spectator seat.
    values.nickname,
    ...values.exclude.split(","),
  ]
    .map((name) => name.trim())
    .filter(Boolean),
);
const periodMs = 1000 / Number(values.hz);
const mapMs = Number(values["map-ms"]);
const engine = await loadEngine({ mod: values.mod });
const spec = observationSpec({
  foeSlots: Number(values["observation-foes"]),
  weaponFeatures: engine.weaponFeatures,
});

const browser = await chromium.connectOverCDP(`http://127.0.0.1:${values["cdp-port"]}`, {
  timeout: 5000,
});
const pages = [];
for (const context of browser.contexts()) {
  for (const page of context.pages()) {
    if (!/webliero\.com/.test(page.url())) continue;
    const inRoom = await page
      .evaluate(() => Boolean(document.querySelector(".game-view")))
      .catch(() => false);
    if (inRoom) pages.push(page);
  }
}
if (!pages.length && values["room-url"]) {
  // Nobody is in a room yet, so open one tab and sit in it. Spectating on
  // purpose: a seat taken is a seat a person cannot have, and there is nothing
  // to be learned from watching this project's own policy play.
  log.info("joining_to_watch", { url: values["room-url"], as: values.nickname });
  const context = browser.contexts()[0];
  const page = await context.newPage();
  page.on("dialog", (dialog) => void dialog.accept().catch(() => {}));
  await joinRoom(page, values["room-url"], { nickname: values.nickname });
  pages.push(page);
}
if (!pages.length) {
  console.error(
    "No tab is in a room. Join one and run this again, or pass --room-url to watch one.",
  );
  process.exit(1);
}
// One tab is enough: it can see everybody.
const observer = new WebLieroObserver(pages[0], { log });

const idleMs = Number(values["idle-seconds"]) * 1000;
const minPlayers = Number(values["min-players"]);
const idleRun = Number(values["idle-run"]);
/** Consecutive samples each player has pressed nothing for. */
const idleFor = new Map();
const minSamples = Number(values["min-samples"]);

let started = new Date();
let id = started.toISOString().replace(/[:.]/g, "-").replace("Z", "");
const directory = new URL(
  values.out ? `${values.out}/` : "../../artifacts/demos/",
  values.out ? `file://${process.cwd()}/` : import.meta.url,
);
await mkdir(directory, { recursive: true });
const RECORD_BYTES = spec.vectorSize * 4 + PATCH_CELLS + MAP_SIZE + ACTION_HEADS.length + 1;
let sink = createWriteStream(new URL(`${id}.bin`, directory));
const describe = () =>
  writeFile(
  new URL(`${id}.json`, directory),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      startedAt: started.toISOString(),
      vectorSize: spec.vectorSize,
      foeSlots: spec.foeSlots,
      patchCells: PATCH_CELLS,
      patchShape: PATCH_SHAPE,
      mapCells: MAP_SIZE,
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      recordBytes: RECORD_BYTES,
      order: ["vector:f32", "patch:u8", "map:u8", "heads:u8", "playerId:u8"],
      engineSha256: engine.sha256,
      mod: engine.settings.name,
      hz: Number(values.hz),
    },
    null,
    2,
  )}\n`,
  );
await describe();
console.log(
  `recording to ${directory.pathname}${id}.bin\n` +
    (values.learn
      ? `learning starts on its own once nobody has played for ${values["idle-seconds"]}s\n`
      : "") +
    "(Ctrl+C to stop)",
);

const scratch = {
  vector: new Float32Array(spec.vectorSize),
  patchBytes: new Uint8Array(PATCH_CELLS),
  map: new Uint8Array(MAP_SIZE),
};
const record = Buffer.alloc(RECORD_BYTES);
const previous = new Map();
const counts = new Map();
let terrain = null;
let terrainAt = 0;
let mapTerrain = null;
let samples = 0;

/** What the player did that is not in the key bitmask. */
function messages(player, worm) {
  const before = previous.get(player.id);
  const now = { rope: Boolean(worm.rope), weapon: worm.selectedWeapon };
  previous.set(player.id, now);
  if (!before) return { rope: 0, weapon: 0 };
  const slots = worm.weapons.length || 5;
  const moved = (now.weapon - before.weapon + slots) % slots;
  return {
    rope: now.rope && !before.rope ? 1 : !now.rope && before.rope ? -1 : 0,
    // One step either way is a key press; anything else is a respawn.
    weapon: moved === 1 ? 1 : moved === slots - 1 ? -1 : 0,
  };
}

/**
 * Whether the room is running the game these observations are encoded for.
 *
 * Checked once the first snapshot arrives, and then never again — the mod
 * cannot change under a room. Worth checking at all because the failure is
 * silent: the same 144 numbers come out either way, and only the weapon block
 * is quietly describing somebody else's weapons.
 */
let roomMod = null;

function roomIsOurs(game) {
  if (roomMod !== null) return roomMod;
  const theirs = game.room?.mod ?? null;
  const ours = engine.settings.name;
  roomMod = theirs === null || theirs === ours;
  if (!roomMod) {
    console.error(
      `This room is running ${theirs} and these observations are encoded for ` +
        `${ours}. The weapons are not the same weapons, so the recording would ` +
        `teach the wrong ones. Re-run with --mod for ${theirs}, or watch another room.`,
    );
    process.exit(1);
  }
  log.info("room_mod", { mod: theirs ?? "unreported", encoding: ours });
  return roomMod;
}

async function sample() {
  const now = Date.now();
  const read = await observer.read().catch(() => null);
  if (!read?.game) return;
  if (!roomIsOurs(read.game)) return;
  if (!terrain || now - terrainAt > mapMs) {
    const map = await observer.read({ terrain: true }).catch(() => null);
    if (map?.game) {
      terrain = map.game;
      terrainAt = now;
      mapTerrain = null;
    }
  }
  if (!terrain) return;
  // Who is actually playing, before anything is written. Spectators have no
  // worm, and a worm waiting to respawn has none either, so both drop out here.
  const playing = read.game.players.filter(
    (player) =>
      player.alive &&
      player.worm &&
      !excluded.has(player.name) &&
      // A replicated worm that has never carried an input cannot be learned from.
      Number.isFinite(player.worm.keys),
  );
  // A room is empty most of the time, and one person alone in it is not playing
  // the game this is trying to learn — there is nobody to fight, so what gets
  // recorded is wandering. Wait for a match.
  if (playing.length < minPlayers) return;
  for (const player of playing) {
    // Somebody standing still is still standing still whether they are away or
    // thinking, and three seconds of it is the same frame forty-five times. A
    // short pause is part of play and stays; a long one stops being recorded
    // until they touch something again.
    const keys = player.worm.keys;
    const still = keys === 0 ? (idleFor.get(player.id) ?? 0) + 1 : 0;
    idleFor.set(player.id, still);
    if (still > idleRun) continue;
    const view = viewFromSnapshot(read.game, terrain, { playerId: player.id });
    if (!mapTerrain) mapTerrain = encodeMapTerrain(view.terrain);
    observe(view, scratch, ["vector", "patchBytes", "map"], spec, mapTerrain);
    encodeMap(view, mapTerrain, scratch.map);
    const heads = headsFromKeys(player.worm.keys, messages(player, player.worm));
    let at = 0;
    for (let index = 0; index < spec.vectorSize; index++) {
      record.writeFloatLE(scratch.vector[index], at);
      at += 4;
    }
    at += Buffer.from(scratch.patchBytes.buffer, scratch.patchBytes.byteOffset, PATCH_CELLS).copy(record, at);
    at += Buffer.from(scratch.map.buffer, scratch.map.byteOffset, MAP_SIZE).copy(record, at);
    for (const choice of heads) record.writeUInt8(choice, at++);
    record.writeUInt8(player.id & 0xff, at);
    sink.write(Buffer.from(record));
    samples++;
    lastSampleAt = Date.now();
    counts.set(player.name, (counts.get(player.name) ?? 0) + 1);
  }
}

/**
 * Play stops when the person stops — going to spectate, leaving, or just
 * putting it down — and nobody should have to say so. A stretch with nothing
 * recorded closes the file and starts the learning, and whatever is played
 * next goes into a new one.
 */
let lastSampleAt = Date.now();
let segmentStart = 0;
let learning = null;

async function finishSegment() {
  const captured = samples - segmentStart;
  if (captured < minSamples) {
    console.log(
      `nobody has played for a while, and ${captured} samples is too few to learn from — still listening`,
    );
    segmentStart = samples;
    return;
  }
  const finished = id;
  await new Promise((resolve) => sink.end(resolve));
  console.log(`\n${captured} samples in ${finished}; learning from it now`);
  // The next stretch of play goes somewhere new, so this one can be trained on
  // while it is still being recorded.
  started = new Date();
  id = started.toISOString().replace(/[:.]/g, "-").replace("Z", "");
  sink = createWriteStream(new URL(`${id}.bin`, directory));
  await describe();
  segmentStart = samples;
  if (!values.learn) return;
  if (learning && learning.exitCode === null) {
    console.log("(still learning from the last stretch; this one waits its turn)");
    return;
  }
  learning = spawn(process.execPath, ["scripts/clone.js", "--only", finished], {
    cwd: new URL("../../", import.meta.url).pathname,
    stdio: "inherit",
  });
  learning.on("exit", (code) => {
    console.log(code === 0 ? "learning finished" : `learning stopped (${code})`);
  });
}

let stopping = false;
const idle = setInterval(() => {
  if (stopping || samples === segmentStart) return;
  if (Date.now() - lastSampleAt < idleMs) return;
  void finishSegment().catch((error) => log.warn("finish_failed", { message: error.message }));
}, 1000);
idle.unref?.();

const timer = setInterval(() => {
  if (stopping) return;
  void sample().catch((error) => log.warn("sample_failed", { message: error.message }));
}, periodMs);

let reported = 0;
const status = setInterval(() => {
  if (samples === reported) return;
  reported = samples;
  console.log(
    `${samples} samples | ` +
      [...counts.entries()].map(([name, count]) => `${name} ${count}`).join(", "),
  );
}, 5000);

const stop = async () => {
  if (stopping) return;
  stopping = true;
  clearInterval(timer);
  clearInterval(status);
  clearInterval(idle);
  await new Promise((resolve) => sink.end(resolve));
  console.log(`\n${samples} samples from ${counts.size} player(s):`);
  for (const [name, count] of counts) console.log(`  ${name.padEnd(20)} ${count}`);
  console.log(`-> ${directory.pathname}${id}.bin`);
  // Never close a browser reached over CDP: the room dies with it.
  browser.close().catch(() => {});
  process.exit(0);
};
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, stop);
