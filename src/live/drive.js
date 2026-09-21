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
import { writeFile } from "node:fs/promises";
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
  spectate,
} from "../room.js";
import { ACTION_HEADS, actionFromHeads } from "../env/actions.js";
import { loadEngine } from "../env/engine.js";
import {
  MAP_SIZE,
  encodeMapTerrain,
  observationSpec,
  observe,
} from "../env/observation.js";
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
// The observation a policy trained on, not a default one. The engine is loaded
// for its measured weapon profile: without it every weapon feature in the
// vector is zero, which is 72 numbers the policy has never seen at zero, and
// the patch has to be cut at the scale the checkpoint was trained at or it
// reshapes into nonsense. Both used to be left at defaults here, which is a
// policy meeting a different world on its first real match.
const engine = await loadEngine(config.engine);
const spec = observationSpec({
  foeSlots,
  weaponFeatures: engine.weaponFeatures,
  patchScale: config.patchScale ?? 2,
});
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
/**
 * Tabs this driver has driven before, and only those.
 *
 * Taking over "any tab already in a room" was written when the room was ours
 * and every tab in it was one we opened. This room is somebody else's and the
 * person who owns this browser plays in it — and a tab of theirs duly got
 * taken over, so their worm started walking around on its own while ours stood
 * still. A tab is fair game only if it was marked by a driver.
 */
const drivenTab = async (page) =>
  Boolean(await page.evaluate(() => window.__wormyDriver === true).catch(() => false));
const inRoomPages = (await findGamePages(browser)).filter((found) => found.inGame);
const existing = [];
for (const found of inRoomPages) {
  if (await drivenTab(found.page)) existing.push(found);
}
const reused = config.fresh ? [] : existing.slice(0, players);
if (reused.length) say(`taking over ${reused.length} tab(s) already in a room`);

const seats = [];
for (let index = 0; index < players; index++) {
  // Tabs rather than windows: three of these are easier to keep track of in one
  // window, and nothing here needs them side by side.
  const page = reused[index]?.page ?? (await openPage(browser));
  handleDialogs(page);
  const nickname =
    config.nicknames?.[index] ??
    `${config.nickname ?? "Wormy"} ${String.fromCharCode(65 + index)}`;
  if (!reused[index]) {
    await page.goto(LOBBY, { waitUntil: "domcontentloaded", timeout: 30_000 });
    await setNickname(page, nickname, { colour: config.colours?.[index] ?? null }).catch(
      () => {},
    );
  }
  // Claim the tab, so a later run knows this one is ours and every other one
  // belongs to somebody who is playing.
  await page.evaluate(() => {
    window.__wormyDriver = true;
  }).catch(() => {});
  seats.push({
    index,
    page,
    nickname,
    playerId: null,
    controls: null,
    observer: null,
    terrain: null,
    terrainAt: 0,
  });
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
    // Seats for the driven worms and for anyone who wants to join and play.
    maxPlayers: config.roomSize ?? 20,
    isPublic: config.isPublic ?? false,
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
const patches = new Uint8Array(players * spec.patch.cells);
const maps = new Uint8Array(players * MAP_SIZE);
const scratch = seats.map((_, index) => ({
  vector: vectors.subarray(index * spec.vectorSize, (index + 1) * spec.vectorSize),
  patchBytes: patches.subarray(index * spec.patch.cells, (index + 1) * spec.patch.cells),
  map: maps.subarray(index * MAP_SIZE, (index + 1) * MAP_SIZE),
}));
// The level is the same for every seat and changes only where somebody digs,
// so it is reduced once per terrain read rather than once per worm per step.
let mapTerrain = null;

/**
 * One player's observation, built from what the adapter reports.
 *
 * The terrain is read on a slower clock than the state: it is the whole level
 * at a byte a pixel, and it only changes where somebody is digging. The state
 * is read every decision, because that is where the worms are.
 */
/** The room as the first seat last saw it: who is in it, and whether it ended. */
let room = null;

/**
 * Who this process is driving, published for the recorder to skip.
 *
 * The recorder used to be told the names by hand, and the day the worms were
 * renamed it went on excluding the old ones — which would have filed the
 * policy's own play as a person's. Saying it here means the two cannot drift.
 */
const DRIVEN = new URL("../../artifacts/players.json", import.meta.url);
async function publishDriven(playing) {
  await writeFile(
    DRIVEN,
    `${JSON.stringify(
      {
        schemaVersion: 1,
        at: new Date().toISOString(),
        names: seats.map((seat) => seat.nickname),
        playing,
      },
      null,
      2,
    )}\n`,
  ).catch(() => {});
}

/**
 * Whether the room is running the game the policy trained under.
 *
 * Checked once the first snapshot arrives. The failure is silent otherwise:
 * the same vector comes out either way, and only the weapon ids and features
 * are quietly describing somebody else's weapons.
 */
let roomMod = null;
function roomIsOurs(game) {
  if (roomMod !== null) return roomMod;
  const theirs = game.room?.mod ?? null;
  const ours = engine.settings.name;
  roomMod = theirs === null || theirs === ours;
  if (!roomMod) {
    say(
      `This room is running ${theirs} and the policy trained under ${ours}. ` +
        "The weapons are not the same weapons, so it would be playing a game it " +
        "never learned. Train a policy for this mod, or find a room running the other.",
    );
    process.exit(1);
  }
  log.info("room_mod", { mod: theirs ?? "unreported", trained: ours });
  return roomMod;
}

async function look(seat, now) {
  const read = await seat.observer.read().catch(() => null);
  if (seat.index === 0 && read?.game) room = read.game;
  if (read?.game) roomIsOurs(read.game);
  // Which player this tab is, by the id the game gave it. Names are not ours
  // alone: somebody in this room is playing under the watcher's name right now.
  if (read?.game?.localPlayerId != null) seat.playerId = read.game.localPlayerId;
  if (!read?.game || !spawned(read)) return null;
  if (!seat.terrain || now - seat.terrainAt > mapMs) {
    const map = await seat.observer.read({ terrain: true }).catch(() => null);
    if (map?.game) {
      seat.terrain = map.game;
      seat.terrainAt = now;
      if (seat.index === 0) mapTerrain = null;
    }
  }
  if (!seat.terrain) return null;
  return viewFromSnapshot(read.game, seat.terrain, { lastAction: seat.lastAction ?? null });
}

writeFrame(
  Buffer.from(
    JSON.stringify({
      envs: 1,
      agents: players,
      vectorSize: spec.vectorSize,
      patchCells: spec.patch.cells,
      patchShape: spec.patch.shape,
      mapCells: MAP_SIZE,
      mapShape: [4, 32, 32],
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
  maps.fill(0);
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
      if (!mapTerrain) mapTerrain = encodeMapTerrain(view.terrain);
      observe(view, scratch[seat.index], ["vector", "patchBytes", "map"], spec, mapTerrain);
    }),
  );
  writeFrame(vectors, patches, maps);
}

// ---------------------------------------------------------------------------
// Saying something.
//
// The room is public, so there are people in it who did not set it up and have
// no idea what the three identical worms are. Two things get said: hello and
// what this is, when somebody arrives, and "G G" when a match ends.
//
// Chat is the game's own text box. The worm's keys are let go first — a held
// arrow key would otherwise stay held while the text box has focus, and the
// worm walks into a wall through the whole sentence.

// Nothing on arrival. A bot that introduces itself to a room full of people
// playing is a bot talking about itself, and it was asked to stop.
const GREETING = (config.greeting ?? "").split("|").map((line) => line.trim()).filter(Boolean);
const FAREWELL = (config.farewell ?? "G G").split("|").map((line) => line.trim()).filter(Boolean);
/** Not more than one greeting this often, however many people come and go. */
const GREET_EVERY_MS = 5 * 60 * 1000;

async function say_in_chat(seat, lines) {
  const page = seat.page;
  // Through the keyboard, the way `Controls` does and the way a player does.
  // The chat box sits below the game view and is often off-screen, so clicking
  // it fails with "element is outside of the viewport" — which is exactly what
  // it did. Enter opens it wherever it is; `keys.js` has always had Enter bound
  // to Chat for this reason.
  await seat.controls?.release().catch(() => {});
  const box = page.locator("[data-hook='input']").first();
  const closed = async () =>
    !(await box.evaluate((el) => el === document.activeElement).catch(() => false));
  try {
    for (const line of lines) {
      await page.keyboard.press("Enter");
      await page.waitForTimeout(150);
      // `fill` puts the whole line in at once, and only works once the box is
      // really focused; typing it is the fallback when it is not.
      const filled = await box
        .fill(line, { timeout: 1500, force: true })
        .then(() => true)
        .catch(() => false);
      if (!filled) await page.keyboard.type(line, { delay: 15 });
      await page.keyboard.press("Enter");
      await page.waitForTimeout(250);
    }
  } catch (error) {
    log.warn("chat_failed", { seat: seat.index, message: error.message });
  }
  // However that went, the box must not be left with the focus. While it has
  // it, every arrow key the policy asks for is typed into a text field instead
  // of reaching the worm — which is one worm standing still for the rest of the
  // match while the other two play, and it is what happened.
  for (let attempt = 0; attempt < 3 && !(await closed()); attempt++) {
    await page.keyboard.press("Escape").catch(() => {});
    await page.waitForTimeout(100);
  }
  if (!(await closed())) {
    log.warn("chat_box_stuck_open", { seat: seat.index });
    return false;
  }
  return true;
}

let known = new Set();
let greetedAt = 0;
let matchWasOver = false;
let talking = false;

/**
 * One pass over the room state: greet an arrival, and mark the end of a match.
 *
 * Both are edges, not states — a match that has ended stays ended for as long
 * as the scoreboard is up, and saying "G G" once a frame for ten seconds is
 * how a bot gets kicked from a public room.
 */
async function chatter() {
  if (!room || talking) return;
  const ours = new Set(seats.map((seat) => seat.nickname));
  const here = new Set();
  let arrived = false;
  for (const player of room.players ?? []) {
    here.add(player.id);
    if (!known.has(player.id) && !ours.has(player.name)) arrived = true;
  }
  const first = known.size === 0;
  known = here;

  const ended = Boolean(room.match?.ended);
  const justEnded = ended && !matchWasOver;
  matchWasOver = ended;

  // The first look sees everybody at once, including our own worms starting
  // up; that is not somebody arriving.
  const now = Date.now();
  const greet =
    GREETING.length > 0 && arrived && !first && now - greetedAt > GREET_EVERY_MS;
  if (!greet && !justEnded) return;

  talking = true;
  try {
    // Whichever of ours is alive, so the line does not come from a corpse.
    const speaker = seats.find((seat) => alive[seat.index]) ?? seats[0];
    if (justEnded && FAREWELL.length) await say_in_chat(speaker, FAREWELL);
    if (greet && (await say_in_chat(speaker, GREETING))) greetedAt = now;
  } finally {
    talking = false;
  }
}

// ---------------------------------------------------------------------------
// Taking a seat, and giving it back.
//
// A quiet room is more fun with somebody in it and a busy one is not: the point
// of sitting down is to give whoever is there an opponent, and the moment there
// are enough people the seat is worth more to them than to us. So the worms
// play while the room is short of players and spectate as soon as it is not.

/** At least this long between sitting down and standing up again. */
const SEATING_PAUSE_MS = 20_000;

/** How many people the room should hold before the worms give their seats up. */
const yieldTo = Number(config.yieldTo ?? 0);
/**
 * How many people are actually playing, not counting anything we drive.
 *
 * By what they are doing, not by what they are called. Spectators are already
 * out — they are on no team and have no worm — so there is nothing to exclude
 * by name, and excluding one meant that the person watching, who then sat down
 * and played under the same name, counted as nobody and the seat was never
 * given up. Only our own worms are known by name here, and only because we
 * named them ourselves a moment ago.
 */
function playersPresent() {
  const ours = new Set(seats.map((seat) => seat.nickname));
  const mine = new Set(seats.map((seat) => seat.playerId).filter((id) => id != null));
  let count = 0;
  for (const player of room?.players ?? []) {
    if (mine.size ? mine.has(player.id) : ours.has(player.name)) continue;
    // On a team at all, whether or not their worm is alive this second.
    if (player.team > 0 || player.alive) count++;
  }
  return count;
}

let seating = false;
let seatedSaid = null;

/**
 * Seat or unseat the driven worms to leave room for people.
 *
 * Only ever one at a time, and never while the frame loop is mid-decision —
 * joining a team walks through the game's own menus, which takes seconds.
 */
let seatedAt = 0;

async function takeSeats() {
  if (seating || !room) return;
  // Joining and spectating both walk the game's menus and take seconds, and the
  // count flickers as people die and respawn. Without a pause between changes
  // the worms spend the match standing up and sitting down.
  if (Date.now() - seatedAt < SEATING_PAUSE_MS) return;
  const people = playersPresent();
  const wanted = Math.max(0, Math.min(seats.length, yieldTo - people));
  // Seated according to the game, not according to whether a panel is on
  // screen: that panel is in the page whether or not anybody is spectating,
  // so reading it had the worms convinced they had already stood up.
  const teams = new Map((room.players ?? []).map((player) => [player.id, player.team]));
  const seated = seats.filter(
    (seat) => seat.playerId != null && (teams.get(seat.playerId) ?? 0) > 0,
  );
  if (seated.length === wanted) return;
  seating = true;
  try {
    if (seated.length > wanted) {
      for (const seat of seated.slice(wanted)) {
        await seat.controls?.release().catch(() => {});
        const left = await spectate(seat.page).catch((error) => {
          log.warn("spectate_failed", { seat: seat.index, message: error.message });
          return false;
        });
        if (!left) log.warn("still_seated", { seat: seat.index, nickname: seat.nickname });
      }
    } else {
      for (const seat of seats.filter((s) => !seated.includes(s)).slice(0, wanted - seated.length)) {
        await joinTeam(seat.page, "any", { timeoutMs: 8000 }).catch(() => {});
      }
    }
    seatedAt = Date.now();
    const now = `${wanted} of ${seats.length} playing, ${people} people in the room`;
    if (now !== seatedSaid) {
      say(now);
      seatedSaid = now;
    }
    await publishDriven(wanted);
  } finally {
    seating = false;
  }
}

// Said before the first frame, not only when the seating changes: the recorder
// needs the names from the moment there is anything to record.
await publishDriven(seats.length);
// On a timer, not only when the seating changes. The recorder ignores this
// file once it is a minute old — a stale one must not go on hiding a person
// who shares the name — and a driver that simply keeps playing never changed
// the seating, so the file went stale underneath it and its own worms started
// being filed as somebody's play.
const sayingWhoWeAre = setInterval(
  () => void publishDriven(seats.length).catch(() => {}),
  10_000,
);
sayingWhoWeAre.unref?.();

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
  // One keyboard per tab, and chat borrows it. Pressing Enter to open the chat
  // box while `apply` is holding ArrowLeft hands the held key to a text field.
  // The frame still has to go out: the policy is waiting on one, and a decision
  // skipped is a worm that coasts, while a frame skipped is a hung run.
  if (talking || seating) {
    await sample();
    return;
  }
  await Promise.all(
    seats.map(async (seat) => {
      if (!alive[seat.index]) return;
      const action = actionFromHeads(heads, seat.index * HEADS);
      // Remembered before it is applied: it is the decision the policy sees
      // next time, whether or not the keyboard took it.
      seat.lastAction = action;
      await seat.controls.apply(action).catch((error) => {
        log.warn("apply_failed", { seat: seat.index, message: error.message });
      });
    }),
  );
  await sample();
  void chatter().catch((error) => log.warn("chatter_failed", { message: error.message }));
  void takeSeats().catch((error) => log.warn("seating_failed", { message: error.message }));
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
