// A page of its own for watching a training run.
//
// Separate from the game dashboard on purpose: that one draws the world as it is
// right now, this one draws how the last few hours have gone, and they are read
// at completely different moments. Different port, different page, no shared
// state — a training run does not need a browser open, and the dashboard does
// not need a run.
//
// It only reads the run directory. Nothing here can start, stop or change a run,
// which is what makes it safe to leave open.
import { createServer } from "node:http";
import { createViewer } from "./viewer.js";
import { pinSeed } from "./seed.js";
import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  DEFAULT_RUNS_DIR,
  describeRun,
  listRuns,
  readRecords,
} from "./recorder.js";

const MAX_CLIENTS = 16;
const WATCH_PORT = 8769;
const WATCHER = fileURLToPath(new URL("../../scripts/watch.js", import.meta.url));
const BLOCKED_CLIENT_MS = 15_000;
const POLL_MS = 500;
// Charts do not need more than this, and a run left going overnight should not
// grow the server's memory without limit.
const MAX_CACHED_RECORDS = 20_000;

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

export async function createMonitorServer({
  port = 8767,
  dir = DEFAULT_RUNS_DIR,
  // Loopback by default: on a laptop this page is a window onto your own
  // machine. In a pod it has to bind the pod's interface and answer to the
  // hostname a browser out there actually asks for, which is not the one it
  // bound. The host check stays in both cases — it is what stops a page on
  // another site from driving this one through a browser that can reach it.
  host = "127.0.0.1",
  publicOrigin = null,
  // What the Watch button's viewer should bind and be reached at. A viewer
  // that binds loopback inside a pod is a viewer nobody outside can open.
  viewerHost = null,
  viewerOrigin = null,
  viewerBasePath = "",
  viewerScript = WATCHER,
  // A path this is mounted under, such as "/ai-worm". No trailing slash.
  basePath = "",
  seedId = process.env.WORMY_SEED_ID,
  seedRun = process.env.WORMY_SEED_RUN,
} = {}) {
  if (Boolean(seedId) !== Boolean(seedRun)) throw new Error("Seed ID and source run are both required");
  const seed = seedId ? await pinSeed(dir, seedId, seedRun) : null;
  if (seed) console.log(`seed | ${seed.id} | sha256 ${seed.sha256}`);
  const base = basePath.replace(/\/+$/, "");
  const assets = new Map(
    await Promise.all(
      [
        ["/", "index.html", "text/html; charset=utf-8"],
        ["/app.js", "app.js", "text/javascript; charset=utf-8"],
        ["/style.css", "style.css", "text/css; charset=utf-8"],
      ].map(async ([path, file, type]) => [
        path,
        {
          body: await readFile(new URL(`../../public/train/${file}`, import.meta.url)),
          type,
        },
      ]),
    ),
  );
  const clients = new Set();
  // The one thing on this server that is not a read. A page of numbers cannot
  // tell you whether a policy looks like someone playing, so there is a button
  // that puts the best one on a map and shows it.
  const viewer = createViewer({
    dir,
    script: viewerScript,
    args: [
      "--port", String(WATCH_PORT),
      ...(viewerHost ? ["--host", viewerHost] : []),
      ...(viewerOrigin ? ["--public-origin", viewerOrigin] : []),
      ...(viewerBasePath ? ["--base-path", viewerBasePath] : []),
    ],
  });
  // One follower per run, shared by every client watching it, so a long run is
  // read from disk once per poll and not once per browser tab.
  const followers = new Map();
  let runsSignature = "";
  let closing = false;
  let origin;

  const follower = (id) => {
    let found = followers.get(id);
    if (!found) {
      found = { id, bytes: 0, records: [], dropped: 0 };
      followers.set(id, found);
    }
    return found;
  };

  const advance = async (id) => {
    const state = follower(id);
    const { records, bytes } = await readRecords(dir, id, state.bytes);
    state.bytes = bytes;
    if (records.length) {
      state.records.push(...records);
      const excess = state.records.length - MAX_CACHED_RECORDS;
      if (excess > 0) {
        state.records.splice(0, excess);
        state.dropped += excess;
      }
    }
    return records.length;
  };

  const write = (client, event, data) => {
    if (client.blocked || client.response.destroyed) return;
    if (!client.response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)) {
      // A tab that cannot keep up misses records rather than growing a buffer;
      // one that stays stuck is dropped by the poll below.
      client.blocked = true;
      client.blockedAt = Date.now();
    }
  };

  const resolveRun = async (wanted) => {
    if (wanted && wanted !== "latest") return wanted;
    const runs = await listRuns(dir);
    return runs[0]?.id ?? null;
  };

  const poll = async () => {
    if (!clients.size) return;
    const runs = await listRuns(dir);
    const signature = JSON.stringify(
      runs.map((run) => [run.id, run.status, run.updatedAt, run.bytes]),
    );
    const changed = signature !== runsSignature;
    runsSignature = signature;
    const watched = new Set([...clients].map((client) => client.runId).filter(Boolean));
    for (const id of watched) await advance(id);
    for (const id of followers.keys()) if (!watched.has(id)) followers.delete(id);
    for (const client of clients) {
      if (client.blocked) {
        if (Date.now() - client.blockedAt > BLOCKED_CLIENT_MS) client.response.destroy();
        continue;
      }
      if (changed) write(client, "runs", { runs });
      if (!client.runId) continue;
      const state = follower(client.runId);
      const fresh = state.records.slice(
        Math.max(0, client.sent - state.dropped),
      );
      if (fresh.length) {
        client.sent = state.dropped + state.records.length;
        write(client, "records", { run: client.runId, records: fresh });
      }
      if (changed) {
        const run = runs.find((one) => one.id === client.runId);
        if (run) write(client, "run", { run });
      }
      if (!fresh.length && !changed && !client.response.write(": heartbeat\n\n")) {
        client.blocked = true;
        client.blockedAt = Date.now();
      }
    }
  };

  const server = createServer(async (request, response) => {
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
    if (closing) return json(503, { error: "Server is shutting down" });
    if (!meantForUs(request, publicOrigin)) {
      return json(403, { error: "Use the monitor's own origin" });
    }
    const url = new URL(request.url, origin);
    // Everything below routes as though it were mounted at the root.
    const slash = needsSlash(url.pathname, base);
    if (slash) {
      response.writeHead(308, { Location: `${slash}${url.search}` });
      return response.end();
    }
    const routed = underBase(url.pathname, base);
    if (routed === null) return json(404, { error: "Not found" });
    url.pathname = routed;
    if (request.method === "POST") {
      const watch = url.pathname.match(/^\/runs\/([A-Za-z0-9_.-]+)\/watch$/);
      if (watch) {
        const started = await viewer.start(watch[1]);
        return started.error ? json(500, started) : json(200, started);
      }
      if (url.pathname === "/watch/stop") {
        await viewer.stop();
        return json(200, { stopped: true });
      }
    }
    if (request.method !== "GET") {
      response.setHeader("Allow", "GET, POST");
      return json(405, { error: "Only GET, and POST to start or stop a viewer" });
    }
    try {
      if (seed && url.pathname === `/seeds/${seed.id}.pt`) {
        response.writeHead(200, {
          "Content-Type": "application/octet-stream",
          "Content-Length": seed.bytes.length,
          "X-Checkpoint-SHA256": seed.sha256,
        });
        return response.end(seed.bytes);
      }
      if (url.pathname === "/health") {
        const runs = await listRuns(dir);
        return json(200, {
          ok: true,
          runs: runs.length,
          running: runs.filter((run) => run.status === "running").length,
          clients: clients.size,
        });
      }
      if (url.pathname === "/watch") {
        const watching = viewer.current;
        const live = watching && watching.child.exitCode === null && watching.child.signalCode === null;
        return json(200, {
          watching: live ? watching.id : null,
          url: live ? watching.reply.url : null,
        });
      }
      // What the room watcher is seeing. It is a separate process writing a
      // separate file, so this is a read of that file and nothing more: no
      // watcher running simply means no file.
      if (url.pathname === "/observer") {
        try {
          const raw = await readFile(new URL("../../artifacts/observer.json", import.meta.url), "utf8");
          const state = JSON.parse(raw);
          // Its own clock decides whether it is live; a file left behind by a
          // watcher that died an hour ago must not read as "watching".
          const age = Date.now() - Date.parse(state.at ?? 0);
          return json(200, { ...state, live: Number.isFinite(age) && age < 15_000, ageMs: age });
        } catch {
          return json(200, { live: false, watching: null });
        }
      }
      if (url.pathname === "/runs") return json(200, { runs: await listRuns(dir) });
      // What `npm run evaluate --history` wrote beside the run's checkpoints,
      // when it has been run: the run's best against its own earlier selves,
      // measured on the field rather than read off the reward.
      const history = url.pathname.match(/^\/runs\/([A-Za-z0-9_.-]+)\/history$/);
      if (history && !history[1].startsWith(".")) {
        const base = dir instanceof URL ? dir : pathToFileURL(`${String(dir).replace(/\/?$/, "/")}`);
        try {
          const raw = await readFile(new URL(`${history[1]}/history.json`, base), "utf8");
          return json(200, JSON.parse(raw));
        } catch {
          return json(404, { error: "No history for this run" });
        }
      }
      if (url.pathname.startsWith("/runs/")) {
        const id = decodeURIComponent(url.pathname.slice("/runs/".length));
        // The id becomes a path segment, so anything that could climb out of the
        // run directory is refused rather than cleaned up.
        if (!/^[A-Za-z0-9_.-]+$/.test(id) || id.startsWith(".")) {
          return json(400, { error: "Not a run id" });
        }
        const run = await describeRun(dir, id);
        if (!run) return json(404, { error: "No such run" });
        const from = Number(url.searchParams.get("from") ?? 0);
        const { records, bytes } = await readRecords(
          dir,
          id,
          Number.isFinite(from) && from > 0 ? from : 0,
        );
        return json(200, { run, records, bytes });
      }
      if (url.pathname === "/events") {
        if (clients.size >= MAX_CLIENTS)
          return json(503, { error: "Too many monitor clients" });
        const runId = await resolveRun(url.searchParams.get("run"));
        response.writeHead(200, {
          "Content-Type": "text/event-stream",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
        });
        response.write("retry: 1000\n\n");
        const client = { response, runId, sent: 0, blocked: false, blockedAt: 0 };
        clients.add(client);
        response.on("close", () => clients.delete(client));
        response.on("error", () => clients.delete(client));
        response.on("drain", () => {
          client.blocked = false;
        });
        const runs = await listRuns(dir);
        write(client, "runs", { runs });
        if (runId) {
          await advance(runId);
          const state = follower(runId);
          client.sent = state.dropped + state.records.length;
          write(client, "run", { run: runs.find((one) => one.id === runId) ?? null });
          write(client, "records", {
            run: runId,
            records: state.records,
            reset: true,
            dropped: state.dropped,
          });
        }
        return;
      }
      const asset = assets.get(url.pathname);
      if (asset) {
        response.writeHead(200, { "Content-Type": asset.type });
        return response.end(asset.body);
      }
      json(404, { error: "Not found" });
    } catch (error) {
      if (!response.headersSent) json(500, { error: error.message });
      else response.destroy();
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });
  origin = `http://127.0.0.1:${server.address().port}`;

  let polling = false;
  const timer = setInterval(() => {
    // Skip a beat rather than stack reads when the disk is slow.
    if (polling) return;
    polling = true;
    poll()
      .catch(() => {})
      .finally(() => {
        polling = false;
      });
  }, POLL_MS);
  timer.unref();
  return {
    origin,
    dir,
    get clients() {
      return clients.size;
    },
    async close() {
      closing = true;
      await viewer.stop();
      clearInterval(timer);
      for (const client of clients) client.response.end();
      const closed = new Promise((resolve) => server.close(resolve));
      server.closeAllConnections();
      await closed;
    },
  };
}
