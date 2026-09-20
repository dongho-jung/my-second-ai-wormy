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
  findGamePages,
  launchDetached,
  openPage,
  prepareProfile,
} from "../browser.js";
import {
  SPECTATING,
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

// stdout is the frame channel, and nothing else may touch it. `createLogger`
// writes info lines with console.log, so one log in a helper puts plain text in
// the middle of a binary stream and the reader waits forever on a frame length
// that was really the letters "[bi". Everything chatty goes to stderr instead.
const toStderr = (...parts) =>
  process.stderr.write(`${parts.map(String).join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;


// A dropped CDP call or a dialog that closed itself should cost one decision,
// not the match. Setup failures still stop, loudly, before any of this matters.
let playing = false;
process.on("unhandledRejection", (error) => {
  if (!playing) {
    process.stderr.write(`\n${error?.stack ?? error}\n`);
    process.exit(1);
  }
  process.stderr.write(`recovered: ${error?.message ?? error}\n`);
});

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
const port = config.cdpPort ?? 9334;
// Attaching to the Chromium the last run left behind keeps the room, and the
// CAPTCHA already solved for it, across restarts. Both of these hand back a
// wrapper; the browser is inside it.
let opened = await attach(port, { timeoutMs: 2000 }).catch(() => null);
if (!opened) {
  await prepareProfile(profile);
  say("starting Chromium");
  opened = await launchDetached({
    port,
    profilePath: profile,
    executablePath: config.browserPath,
    headless: false,
  });
}
const browser = opened.browser;

/**
 * The game asks "are you sure you want to leave the room?" on its way out of
 * one, and anything else a page decides to pop up arrives the same way. Without
 * a handler Playwright dismisses them itself and races its own dismissal, which
 * raises a protocol error from nowhere and takes the process with it. Accepting
 * them here is both what a player does and what keeps a match alive.
 */
function handleDialogs(page) {
  page.on("dialog", (dialog) => void dialog.accept().catch(() => {}));
}

// Tabs already sitting in a room are taken over rather than abandoned. A run
// that ends leaves its players seated, and WebLiero asks for a CAPTCHA to make
// a room — so opening three more tabs beside them would cost a seat each and a
// CAPTCHA, for nothing.
const existing = (await findGamePages(browser)).filter((found) => found.inGame);
const reused = config.fresh ? [] : existing.slice(0, players);
if (reused.length) say(`taking over ${reused.length} tab(s) already in a room`);

const seats = [];
for (let index = 0; index < players; index++) {
  // Tabs rather than windows: three of these are easier to keep track of in one
  // window, and nothing here needs them side by side.
  const page = reused[index]?.page ?? (await openPage(browser));
  handleDialogs(page);
  const nickname = `${config.nickname ?? "Wormy"} ${String.fromCharCode(65 + index)}`;
  if (!reused[index]) {
    await page.goto(LOBBY, { waitUntil: "domcontentloaded", timeout: 30_000 });
    await setNickname(page, nickname).catch(() => {});
  }
  seats.push({ index, page, nickname, controls: null, observer: null, terrain: null, terrainAt: 0 });
}
const seated = reused.length >= players;

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
if (seated) {
  // Everyone is already where they need to be. Creating a room does not change
  // the address bar, so a taken-over tab shows the lobby; the link comes from
  // the game's own dialog, and is worth having to hand for a person to join.
  url = await roomUrl(seats[0].page).catch(() => seats[0].page.url());
  say(`room ${url}`);
} else if (!url) {
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
if (seated) {
  // A taken-over tab may be spectating rather than playing — but most are
  // already in, and waiting thirty seconds for a spectating panel that will
  // never appear costs a minute and a half of startup for nothing.
  for (const seat of seats) {
    const spectating = await seat.page
      .locator(SPECTATING)
      .isVisible()
      .catch(() => false);
    if (spectating) await joinTeam(seat.page, "any", { timeoutMs: 5000 }).catch(() => {});
  }
}

for (const seat of seated ? [] : seats.slice(1)) {
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

playing = true;
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
