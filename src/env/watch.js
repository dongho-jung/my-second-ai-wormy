// One match, played at the speed a person can watch, drawn in a browser.
//
// The numbers on the training page say whether a policy is improving. They do
// not say whether it looks like someone playing. This runs a single match at
// real time and serves it as a picture: the same engine, the same policy, just
// slowed down to sixty ticks a second and pointed at a canvas.
//
// It is the headless engine rendered, not the live game at webliero.com. The
// physics are identical — the same bundle, checked by the same checksum — but
// putting a policy into a real online room needs the key-input path this
// repository still does not have.
//
// Actions arrive on stdin in the same frames `worker.js` uses, so whatever can
// drive a training worker can drive this.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { actionFromHeads, ACTION_HEADS } from "./actions.js";
import { loadEngine } from "./engine.js";
import { readFileSync } from "node:fs";
import { WormEnv } from "./env.js";
import { GhostRace } from "./race.js";
import { MAP, MAP_SIZE } from "./observation.js";
import { gzipSync } from "node:zlib";

const HEADS = ACTION_HEADS.length;
const MAX_CLIENTS = 8;

// stdout is the frame channel, and nothing else may touch it. `createLogger`
// writes info lines with console.log, so one log in a helper puts plain text in
// the middle of a binary stream and the reader waits forever on a frame length
// that was really the letters "[bi". Everything chatty goes to stderr instead.
const toStderr = (...parts) =>
  process.stderr.write(`${parts.map(String).join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;


const config = JSON.parse(process.argv[2] ?? "{}");
const port = config.port ?? 8769;
// What to bind, and what to answer to.
//
// Loopback by default: on a laptop this is a window onto your own machine and
// has no business being reachable from the network. Somewhere else — a pod
// behind an ingress — it has to bind the pod's interface and accept the
// hostname the browser actually asks for, which is not the one it bound.
//
// The host check stays either way. It is what stops a page on another site
// from pointing a script at this viewer through the browser that can reach it,
// so opening the bind address widens what may connect without widening what
// may pretend to be it.
const host = config.host ?? "127.0.0.1";
const publicOrigin = config.publicOrigin ?? null;
// A path this is mounted under, such as "/ai-worm/watch". No trailing slash.
const base = (config.basePath ?? "").replace(/\/+$/, "");
const speed = config.speed ?? 1;
const agents = config.agents ?? 3;

const engine = await loadEngine(config.engine);
// The maps the policy was trained on, cycled the way the vector environment
// cycles them. Without these the viewer generates its own dirt fields, and a
// policy trained on a room's real maps is watched somewhere it has never been —
// which is how the dashboard came to say "Random Dirt" all evening.
const stock = (config.levelFiles ?? []).map((path) => {
  const file = path.split("/").pop();
  return { file, level: engine.readAnyLevel(file, readFileSync(path)) };
});
const levels = new Map();
for (const { file, level } of stock) {
  for (const name of [file, level.name]) {
    levels.set(name, level);
    levels.set(name.toLowerCase(), level);
  }
}
// The whole world the checkpoint carries, then the few things that are about
// watching rather than about the game.
//
// Naming the settings one by one here was the same bug the checkpoint's `world`
// exists to end: every setting added to training had to be copied into this
// list too, and the ones that were missed did not fail — they quietly showed a
// different game. A movement run watched through a hand-written list kept its
// trigger, because `lockWeapons` was not on it.
//
// Anything the environment does not know is ignored, so the viewer's own keys
// riding along is harmless.
const world = {
  ...config,
  ...(stock.length ? { level: (_, seed) => stock[seed % stock.length].level } : {}),
  agents,
  // The page draws all three, whatever the run was trained to hand out.
  observations: ["vector", "patchBytes", "map"],
  seed: config.seed ?? Math.floor(Math.random() * 0xffffffff),
};
const racing = Array.isArray(config.race?.scenarios) && config.race.scenarios.length > 0;
const env = racing
  ? new GhostRace(engine, {
      racers: agents,
      scenarios: config.race.scenarios,
      levels,
      world,
      episodeTicks: config.race.episodeTicks,
    })
  : new WormEnv(engine, world);
if (!racing) env.reset();

/** Below this, packing costs more than it saves. */
const GZIP_OVER = 4096;

let levelVersion = 0;
let episodes = 0;
const scores = Array.from({ length: agents }, () => ({ kills: 0, deaths: 0, damage: 0 }));

const palette = () => Array.from(env.world.level.Ha ?? engine.settings.Ha);

const levelPayload = () => {
  const data = env.world.level.data;
  const chunks = [];
  for (let at = 0; at < data.length; at += 32768) {
    chunks.push(String.fromCharCode(...data.subarray(at, at + 32768)));
  }
  return {
    version: levelVersion,
    name: env.world.level.name,
    width: env.world.level.width,
    height: env.world.level.height,
    encoding: "base64-u8",
    order: "row-major",
    data: Buffer.from(chunks.join(""), "binary").toString("base64"),
    paletteRgb: palette(),
    materialFlags: Array.from(engine.materialFlags),
  };
};

const state = () => {
  const level = env.world.level;
  const race = racing ? env.info() : null;
  const racers = race?.racers ?? env.worms.map((worm, id) => ({
    id,
    worm,
    loadout: env.loadouts[id],
    progress: env.progress[id],
    finish: null,
    rank: null,
  }));
  const projectiles = (racing ? env.envs : [env]).flatMap((one, worldIndex) =>
    [one.world.Ib, one.world.Zb].flatMap((pool) => {
      const out = [];
      for (let slot = 0; slot < pool.$; slot++) {
        const entity = pool.list[slot];
        if (!entity.u) continue;
        out.push({
          x: entity.x,
          y: entity.y,
          owner: racing ? worldIndex : (entity.H < 0 ? null : entity.H),
        });
      }
      return out;
    }),
  );
  return {
    tick: env.world.qb,
    elapsedTicks: race?.elapsedTicks ?? env.info().elapsedTicks,
    episode: env.episode,
    episodes,
    seed: env.episodeSeed,
    speed,
    levelVersion,
    mode: racing ? "ghost-race" : "match",
    race: race
      ? {
          scenario: race.scenario,
          scenarios: race.scenarios,
          start: race.start,
          goal: race.goal,
          detour: race.detour,
          fingerprint: config.race.fingerprint ?? null,
          done: race.done,
          settled: race.settled,
        }
      : null,
    map: { name: level.name, width: level.width, height: level.height },
    worms: racers.map(({ id, worm, loadout, progress, finish, rank, dnf = false }) => ({
      id,
      alive: Boolean(worm.u),
      x: finish?.x ?? worm.x,
      y: finish?.y ?? worm.y,
      health: worm.Xa,
      facing: worm.direction === 1 ? "right" : "left",
      aim: worm.direction === 1 ? -worm.Oa : Math.PI + worm.Oa,
      weapon: worm.O[worm.Ka]?.type.name ?? null,
      ammo: worm.O[worm.Ka]?.ha ?? 0,
      loadout: loadout.map((weapon) => engine.weaponNames[weapon]),
      rope: !finish && worm.Fa.Sc ? { x: worm.Fa.x, y: worm.Fa.y, attached: worm.Fa.jc } : null,
      // Where this worm was told to go, in a run that hands out destinations.
      // Without it the page shows a worm moving and no way to tell whether it
      // is going anywhere on purpose, which is the whole question here.
      goal: race?.goal ?? progress?.goal ?? null,
      ghost: racing,
      policyMode: racing ? (config.race.racerModes?.[id] ?? "sample") : null,
      finishSeconds: finish?.seconds ?? null,
      rank,
      dnf,
      score: scores[id],
    })),
    projectiles,
  };
};

/* --- the page ----------------------------------------------------------- */

const assets = new Map(
  await Promise.all(
    [
      ["/", "index.html", "text/html; charset=utf-8"],
      ["/app.js", "app.js", "text/javascript; charset=utf-8"],
      ["/style.css", "style.css", "text/css; charset=utf-8"],
    ].map(async ([path, file, type]) => [
      path,
      { body: await readFile(new URL(`../../public/watch/${file}`, import.meta.url)), type },
    ]),
  ),
);
const clients = new Set();
let origin;

/**
 * The path this is served under, and the request path with it taken off.
 *
 * Served at `dashboard.example.com/ai-worm/`, every request arrives with that
 * in front of it and none of the routes below know the name. Taking it off
 * here means the routing is the same wherever it is mounted, and the pages ask
 * for their own files by relative path, so they resolve against whatever the
 * document's address turned out to be.
 *
 * Returns null when the prefix is set and the request is not under it — that is
 * somebody else's request arriving on this port.
 */
function underBase(pathname, base) {
  if (!base) return pathname;
  if (pathname === base) return "";            // wants the trailing slash
  if (!pathname.startsWith(`${base}/`)) return null;
  return pathname.slice(base.length) || "/";
}

/** `/ai-worm/` from `/ai-worm`, or null when it is already right. */
function needsSlash(pathname, base) {
  return base && pathname === base ? `${base}/` : null;
}

/**
 * Whether this request is for this viewer, rather than for whatever a page on
 * another site hoped would answer on this port.
 *
 * Both the bound address and the public one count: a pod binds 0.0.0.0 and is
 * reached at an ingress hostname, and both are this viewer.
 */
/**
 * Whether a request's `Host` and `Origin` say it was meant for this server.
 *
 * The check is against DNS rebinding: an attacker's domain resolving to a
 * loopback address still arrives carrying its own name, and is refused. So the
 * question is the name, not the port — and the port is not something this can
 * know anyway. Published on 18768 and listening on 8768, a browser says
 * `localhost:18768` and it is still the same machine.
 *
 * Anything else has to be named, through `--public-origin`.
 */
function loopback(host) {
  if (!host) return false;
  const name = host.startsWith("[") ? host.slice(0, host.indexOf("]") + 1) : host.split(":")[0];
  return name === "127.0.0.1" || name === "localhost" || name === "[::1]";
}

function meantForUs(request, publicOrigin) {
  const host = request.headers.host;
  const named = publicOrigin ? new URL(publicOrigin).host : null;
  if (!(loopback(host) || (named && host === named))) return false;
  const from = request.headers.origin;
  if (!from) return true;
  try {
    return loopback(new URL(from).host) || (publicOrigin && from === publicOrigin);
  } catch {
    return false;
  }
}

function allowed(request) {
  return meantForUs(request, publicOrigin);
}

const server = createServer((request, response) => {
  const json = (status, body) => {
    const text = JSON.stringify(body);
    // The level is the only big answer here and the page asks for it once a
    // second, because the ground is being dug through while they play. It is
    // palette indices with long runs of the same value, so gzip takes 230 KB
    // down to 74 KB and the viewer stops being the heaviest thing on the wire.
    const wants = /\bgzip\b/.test(request.headers["accept-encoding"] ?? "");
    if (wants && text.length > GZIP_OVER) {
      const packed = gzipSync(text);
      response.writeHead(status, {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Encoding": "gzip",
        Vary: "Accept-Encoding",
      });
      return response.end(packed);
    }
    response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    response.end(text);
  };
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
  );
  if (!allowed(request)) {
    return json(403, { error: "Use the viewer's own origin" });
  }
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET");
    return json(405, { error: "Read-only viewer: GET is required" });
  }
  const asked = new URL(request.url, origin).pathname;
  const slash = needsSlash(asked, base);
  if (slash) {
    response.writeHead(308, { Location: slash });
    return response.end();
  }
  const path = underBase(asked, base);
  if (path === null) return json(404, { error: "Not found" });
  if (path === "/health") return json(200, { ok: true, clients: clients.size });
  if (path === "/state") return json(200, state());
  // The terrain is dug through as they play, so it is re-read rather than sent
  // once: this is the picture changing, not a static backdrop.
  if (path === "/level") return json(200, levelPayload());
  if (path === "/events") {
    if (clients.size >= MAX_CLIENTS) return json(503, { error: "Too many viewers" });
    response.writeHead(200, {
      "Content-Type": "text/event-stream",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    response.write("retry: 1000\n\n");
    clients.add(response);
    response.on("close", () => clients.delete(response));
    response.on("error", () => clients.delete(response));
    response.write(`event: state\ndata: ${JSON.stringify(state())}\n\n`);
    return;
  }
  const asset = assets.get(path);
  if (asset) {
    response.writeHead(200, { "Content-Type": asset.type });
    return response.end(asset.body);
  }
  json(404, { error: "Not found" });
});
await new Promise((resolve, reject) => {
  const bind = (on, then) => {
    server.removeAllListeners("error");
    server.once("error", then);
    server.listen(on, host, resolve);
  };
  // A viewer left behind by a monitor that has since restarted still holds the
  // port, and the new one cannot see it to stop it. Rather than fail, take any
  // free port and say which — the address is printed below, and the monitor
  // reads it from there instead of assuming.
  bind(port, (error) => {
    if (error.code !== "EADDRINUSE") return reject(error);
    process.stderr.write(`port ${port} is taken; using another\n`);
    bind(0, reject);
  });
});
origin = `http://127.0.0.1:${server.address().port}`;
// Where somebody can actually reach it, which behind an ingress is not where
// it bound. The monitor reads this line to know what to open, so this is the
// address the Watch button ends up pointing at.
const reachableAt = `${publicOrigin ?? origin}${base}/`;
process.stderr.write(`viewer ${reachableAt}\n`);

/* --- the match ---------------------------------------------------------- */

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

const vectors = new Float32Array(agents * env.spec.vectorSize);
const patches = new Uint8Array(agents * env.spec.patch.cells);
const maps = new Uint8Array(agents * MAP_SIZE);
const resets = new Uint8Array(agents);
const gather = () => {
  for (let agent = 0; agent < agents; agent++) {
    vectors.set(env.observations[agent].vector, agent * env.spec.vectorSize);
    patches.set(env.observations[agent].patchBytes, agent * env.spec.patch.cells);
    maps.set(env.observations[agent].map, agent * MAP_SIZE);
  }
};

writeFrame(
  Buffer.from(
    JSON.stringify({
      envs: 1,
      agents,
      vectorSize: env.spec.vectorSize,
      patchCells: env.spec.patch.cells,
      patchShape: env.spec.patch.shape,
      mapCells: MAP_SIZE,
      mapShape: [MAP.channels.length, MAP.cells, MAP.cells],
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      actionBytes: agents * HEADS,
      restartBytes: agents,
      viewer: reachableAt,
      engineSha256: engine.sha256,
      frameskip: env.frameskip,
      episodeTicks: env.episodeTicks,
    }),
    "utf8",
  ),
);
gather();
resets.fill(1);
writeFrame(vectors, patches, maps, resets);

const broadcast = () => {
  if (!clients.size) return;
  const frame = `event: state\ndata: ${JSON.stringify(state())}\n\n`;
  for (const client of clients) client.write(frame);
};

// One decision is `frameskip` ticks of game time, and the game runs at 60 of
// them a second. Waiting that long between steps is what makes it watchable.
const stepMs = (env.frameskip / 60) * 1000 / speed;
let dueAt = performance.now();
let resetRaceAt = 0;
const RACE_PAUSE_MS = 3000;

const advance = (heads) => {
  resets.fill(0);
  if (racing && env.done) {
    if (performance.now() >= resetRaceAt) {
      episodes++;
      env.reset();
      resets.fill(1);
      levelVersion++;
    }
    gather();
    broadcast();
    writeFrame(vectors, patches, maps, resets);
    return;
  }
  const out = env.step(Array.from({ length: agents }, (_, agent) =>
    actionFromHeads(heads, agent * HEADS),
  ));
  const restarted = out.restarted ?? out.respawned ?? [];
  for (let agent = 0; agent < Math.min(agents, restarted.length); agent++) {
    resets[agent] = restarted[agent] ? 1 : 0;
  }
  if (!racing) {
    for (const [agent, events] of out.info.events.entries()) {
      scores[agent].kills += events.killed;
      scores[agent].deaths += events.died;
      scores[agent].damage += events.damageDealt;
    }
  }
  if (out.done) {
    if (racing) {
      resetRaceAt = performance.now() + RACE_PAUSE_MS;
    } else {
      episodes++;
      env.reset();
      resets.fill(1);
      levelVersion++;
      for (const score of scores) {
        score.kills = 0;
        score.deaths = 0;
        score.damage = 0;
      }
    }
  }
  gather();
  broadcast();
  writeFrame(vectors, patches, maps, resets);
};

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
    const wait = Math.max(0, dueAt - performance.now());
    dueAt = Math.max(performance.now(), dueAt) + stepMs;
    if (wait < 1) advance(heads);
    else setTimeout(() => advance(heads), wait);
  }
});
process.stdin.on("end", () => process.exit(0));
process.stdin.on("error", () => process.exit(0));
