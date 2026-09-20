// The trained policy, playing in a real room on webliero.com.
//
// This is the same shape as `src/env/watch.js` and speaks the same frames, so
// whatever drives a training worker drives this too. What changes is where the
// world is: not a headless engine in this process, but the actual game running
// in browser windows, read over the Chrome DevTools Protocol and driven with
// real key presses.
//
// Nothing here patches the game. The state comes from `snapshotV20`, which only
// reads; the keys go in as events, through the bindings the player set up.
//
// The room is made and joined through the game's own lobby. WebLiero asks for a
// CAPTCHA to create one — that is an anti-automation control on someone else's
// service, so it is reported and waited out in the window, never worked around.
import { parseArgs } from "node:util";
import { WebLieroObserver } from "../observer.js";
import { createLogger } from "../log.js";
import {
  DEFAULT_PROFILE,
  attach,
  launchDetached,
  openWindow,
  prepareProfile,
} from "../browser.js";
import {
  captchaVisible,
  createRoom,
  dismissHowToPlay,
  joinRoom,
  joinTeam,
  roomUrl,
  setNickname,
  spawned,
} from "../room.js";
import { ACTION_HEADS, actionFromHeads } from "../env/actions.js";
import { PATCH_CELLS, observationSpec, observe } from "../env/observation.js";
import { viewFromSnapshot } from "../env/view.js";
import { Controls } from "./controls.js";

const HEADS = ACTION_HEADS.length;
const LOBBY = "https://www.webliero.com/";

const config = JSON.parse(process.argv[2] ?? "{}");
const players = config.players ?? 3;
const decideMs = 1000 / (config.decideHz ?? 15);
const mapMs = config.mapMs ?? 1000;
const foeSlots = config.observationFoes ?? players - 1;
const spec = observationSpec({ foeSlots });
const log = createLogger({ level: config.verbose ? "debug" : "info", scope: "live" });

// stdout carries frames and nothing else; anything to say goes to stderr.
const say = (text) => process.stderr.write(`${text}\n`);

function writeFrame(...parts) {
  let bytes = 0;
  for (const part of parts) bytes += part.byteLength;
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32LE(bytes, 0);
  process.stdout.write(header);
  for (const part of parts) {
    process.stdout.write(Buffer.from(part.buffer, part.byteOffset, part.byteLength));
  }
}

/* --- the windows -------------------------------------------------------- */

const profile = config.profile ?? DEFAULT_PROFILE;
await prepareProfile(profile);
let browser = await attach(config.cdpPort ?? 9334).catch(() => null);
if (!browser) {
  say("starting Chromium");
  await launchDetached({
    port: config.cdpPort ?? 9334,
    profilePath: profile,
    executablePath: config.browserPath,
    headless: false,
  });
  browser = await attach(config.cdpPort ?? 9334, { timeoutMs: 30_000 });
}

const seats = [];
for (let index = 0; index < players; index++) {
  const page = await openWindow(browser, LOBBY);
  const nickname = `${config.nickname ?? "Wormy"} ${String.fromCharCode(65 + index)}`;
  await setNickname(page, nickname).catch(() => {});
  seats.push({ index, page, nickname, controls: null, observer: null, terrain: null, terrainAt: 0 });
}

/** Wait out anything only a person can clear, saying so once. */
async function waitForHuman(page, what) {
  let announced = false;
  for (;;) {
    if (!(await captchaVisible(page))) return announced;
    if (!announced) {
      announced = true;
      say(`\n*** ${what} — please solve the CAPTCHA in the game window. Waiting. ***\n`);
    }
    await page.waitForTimeout(1000);
  }
}

let url = config.roomUrl ?? null;
if (!url) {
  say(`creating a room for ${players}`);
  await createRoom(seats[0].page, {
    name: config.roomName ?? "wormy",
    maxPlayers: players,
    isPublic: false,
    onCaptcha: () =>
      say("\n*** WebLiero is asking for a CAPTCHA to create the room. Please solve it in the game window. ***\n"),
    log,
  });
  await joinTeam(seats[0].page);
  url = await roomUrl(seats[0].page);
  say(`room ${url}`);
} else {
  await joinRoom(seats[0].page, url, { nickname: seats[0].nickname });
  await joinTeam(seats[0].page);
}

for (const seat of seats.slice(1)) {
  await joinRoom(seat.page, url, { nickname: seat.nickname });
  await waitForHuman(seat.page, `${seat.nickname} is being asked to prove it is human`);
  await joinTeam(seat.page);
  await dismissHowToPlay(seat.page).catch(() => {});
}

for (const seat of seats) {
  seat.observer = new WebLieroObserver(seat.page, { log });
  seat.controls = new Controls(seat.page, { log });
  await seat.controls.verify();
  say(`${seat.nickname} ready`);
}

/* --- the loop ----------------------------------------------------------- */

const vectors = new Float32Array(players * spec.vectorSize);
const patches = new Uint8Array(players * PATCH_CELLS);
const scratch = seats.map((_, index) => ({
  vector: vectors.subarray(index * spec.vectorSize, (index + 1) * spec.vectorSize),
  patchBytes: patches.subarray(index * PATCH_CELLS, (index + 1) * PATCH_CELLS),
}));

/**
 * One player's observation, built from what the adapter reports.
 *
 * The terrain is read on a slower clock than the state: it is the whole level
 * at a byte a pixel, and it only changes where somebody is digging. The state
 * is read every decision, because that is where the worms are.
 */
async function look(seat, now) {
  const read = await seat.observer.read().catch(() => null);
  if (!read?.game || !spawned(read)) return null;
  if (!seat.terrain || now - seat.terrainAt > mapMs) {
    const map = await seat.observer.read({ terrain: true }).catch(() => null);
    if (map?.game) {
      seat.terrain = map.game;
      seat.terrainAt = now;
    }
  }
  if (!seat.terrain) return null;
  return viewFromSnapshot(read.game, seat.terrain);
}

writeFrame(
  Buffer.from(
    JSON.stringify({
      envs: 1,
      agents: players,
      vectorSize: spec.vectorSize,
      patchCells: PATCH_CELLS,
      patchShape: [4, 32, 32],
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      actionBytes: players * HEADS,
      live: true,
      room: url,
      frameskip: config.frameskip ?? 4,
      episodeTicks: 0,
    }),
    "utf8",
  ),
);

let alive = new Array(players).fill(false);

async function sample() {
  const now = Date.now();
  vectors.fill(0);
  patches.fill(0);
  await Promise.all(
    seats.map(async (seat) => {
      const view = await look(seat, now);
      alive[seat.index] = Boolean(view);
      if (!view) {
        // Dead, spectating, or the window is showing a dialog. Let go of the
        // keys rather than leaving a worm walking into a wall.
        await seat.controls.release().catch(() => {});
        return;
      }
      observe(view, scratch[seat.index], ["vector", "patchBytes"], spec);
    }),
  );
  writeFrame(vectors, patches);
}

await sample();

let dueAt = Date.now();
let held = Buffer.alloc(0);
process.stdin.on("data", (chunk) => {
  held = held.length ? Buffer.concat([held, chunk]) : chunk;
  for (;;) {
    if (held.length < 4) return;
    const size = held.readUInt32LE(0);
    if (held.length < 4 + size) return;
    const frame = held.subarray(4, 4 + size);
    held = held.subarray(4 + size);
    const heads = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
    const wait = Math.max(0, dueAt - Date.now());
    dueAt = Math.max(Date.now(), dueAt) + decideMs;
    setTimeout(() => void act(heads), wait);
  }
});

async function act(heads) {
  await Promise.all(
    seats.map(async (seat) => {
      if (!alive[seat.index]) return;
      const action = actionFromHeads(heads, seat.index * HEADS);
      await seat.controls.apply(action).catch((error) => {
        log.warn("apply_failed", { seat: seat.index, message: error.message });
      });
    }),
  );
  await sample();
}

const stop = async () => {
  for (const seat of seats) await seat.controls?.release().catch(() => {});
  // Never close a browser attached over CDP: the room dies with it and the next
  // one comes with a CAPTCHA.
  process.exit(0);
};
process.stdin.on("end", stop);
process.stdin.on("error", stop);
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, stop);
