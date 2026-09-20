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
import { WormEnv } from "./env.js";
import { MAP_SIZE, PATCH_CELLS } from "./observation.js";

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
const speed = config.speed ?? 1;
const agents = config.agents ?? 3;

const engine = await loadEngine(config.engine);
const env = new WormEnv(engine, {
  agents,
  episodeTicks: config.episodeTicks ?? 3600,
  frameskip: config.frameskip ?? 4,
  inputLatencyTicks: config.inputLatencyTicks ?? 0,
  observationFoes: config.observationFoes,
  observations: ["vector", "patchBytes", "map"],
  loadout: config.loadout ?? "random",
  seed: config.seed ?? Math.floor(Math.random() * 0xffffffff),
});
env.reset();

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
  return {
    tick: env.world.qb,
    elapsedTicks: env.info().elapsedTicks,
    episode: env.episode,
    episodes,
    seed: env.episodeSeed,
    speed,
    levelVersion,
    map: { name: level.name, width: level.width, height: level.height },
    worms: env.worms.map((worm, agent) => ({
      id: agent,
      alive: Boolean(worm.u),
      x: worm.x,
      y: worm.y,
      health: worm.Xa,
      facing: worm.direction === 1 ? "right" : "left",
      aim: worm.direction === 1 ? -worm.Oa : Math.PI + worm.Oa,
      weapon: worm.O[worm.Ka]?.type.name ?? null,
      ammo: worm.O[worm.Ka]?.ha ?? 0,
      loadout: env.loadouts[agent].map((id) => engine.weaponNames[id]),
      rope: worm.Fa.Sc ? { x: worm.Fa.x, y: worm.Fa.y, attached: worm.Fa.jc } : null,
      score: scores[agent],
    })),
    projectiles: [env.world.Ib, env.world.Zb].flatMap((pool) => {
      const out = [];
      for (let slot = 0; slot < pool.$; slot++) {
        const entity = pool.list[slot];
        if (!entity.u) continue;
        out.push({ x: entity.x, y: entity.y, owner: entity.H < 0 ? null : entity.H });
      }
      return out;
    }),
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

const server = createServer((request, response) => {
  const json = (status, body) => {
    response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    response.end(JSON.stringify(body));
  };
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
  );
  if (
    request.headers.host !== new URL(origin).host ||
    (request.headers.origin && request.headers.origin !== origin)
  ) {
    return json(403, { error: "Use the local viewer origin" });
  }
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET");
    return json(405, { error: "Read-only viewer: GET is required" });
  }
  const path = new URL(request.url, origin).pathname;
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
  server.once("error", reject);
  server.listen(port, "127.0.0.1", resolve);
});
origin = `http://127.0.0.1:${server.address().port}`;
process.stderr.write(`viewer ${origin}\n`);

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
const patches = new Uint8Array(agents * PATCH_CELLS);
const maps = new Uint8Array(agents * MAP_SIZE);
const gather = () => {
  for (let agent = 0; agent < agents; agent++) {
    vectors.set(env.observations[agent].vector, agent * env.spec.vectorSize);
    patches.set(env.observations[agent].patchBytes, agent * PATCH_CELLS);
    maps.set(env.observations[agent].map, agent * MAP_SIZE);
  }
};

writeFrame(
  Buffer.from(
    JSON.stringify({
      envs: 1,
      agents,
      vectorSize: env.spec.vectorSize,
      patchCells: PATCH_CELLS,
      patchShape: [4, 32, 32],
      mapCells: MAP_SIZE,
      mapShape: [4, 32, 32],
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      actionBytes: agents * HEADS,
      viewer: origin,
      engineSha256: engine.sha256,
      frameskip: env.frameskip,
      episodeTicks: env.episodeTicks,
    }),
    "utf8",
  ),
);
gather();
writeFrame(vectors, patches, maps);

const broadcast = () => {
  if (!clients.size) return;
  const frame = `event: state\ndata: ${JSON.stringify(state())}\n\n`;
  for (const client of clients) client.write(frame);
};

// One decision is `frameskip` ticks of game time, and the game runs at 60 of
// them a second. Waiting that long between steps is what makes it watchable.
const stepMs = (env.frameskip / 60) * 1000 / speed;
let dueAt = performance.now();

const advance = (heads) => {
  const before = env.worms.map((worm) => ({ alive: Boolean(worm.u) }));
  const out = env.step(Array.from({ length: agents }, (_, agent) =>
    actionFromHeads(heads, agent * HEADS),
  ));
  for (const [agent, events] of out.info.events.entries()) {
    scores[agent].kills += events.killed;
    scores[agent].deaths += events.died;
    scores[agent].damage += events.damageDealt;
    void before[agent];
  }
  if (out.done) {
    episodes++;
    env.reset();
    levelVersion++;
    for (const score of scores) {
      score.kills = 0;
      score.deaths = 0;
      score.damage = 0;
    }
  }
  gather();
  broadcast();
  writeFrame(vectors, patches, maps);
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
