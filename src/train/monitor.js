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
import { readFile } from "node:fs/promises";
import {
  DEFAULT_RUNS_DIR,
  describeRun,
  listRuns,
  readRecords,
} from "./recorder.js";

const MAX_CLIENTS = 16;
const BLOCKED_CLIENT_MS = 15_000;
const POLL_MS = 500;
// Charts do not need more than this, and a run left going overnight should not
// grow the server's memory without limit.
const MAX_CACHED_RECORDS = 20_000;

export async function createMonitorServer({
  port = 8767,
  dir = DEFAULT_RUNS_DIR,
} = {}) {
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
    if (
      request.headers.host !== new URL(origin).host ||
      (request.headers.origin && request.headers.origin !== origin)
    ) {
      return json(403, { error: "Use the local monitor origin" });
    }
    if (request.method !== "GET") {
      response.setHeader("Allow", "GET");
      return json(405, { error: "Read-only monitor: GET is required" });
    }
    const url = new URL(request.url, origin);
    try {
      if (url.pathname === "/health") {
        const runs = await listRuns(dir);
        return json(200, {
          ok: true,
          runs: runs.length,
          running: runs.filter((run) => run.status === "running").length,
          clients: clients.size,
        });
      }
      if (url.pathname === "/runs") return json(200, { runs: await listRuns(dir) });
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
    server.listen(port, "127.0.0.1", resolve);
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
      clearInterval(timer);
      for (const client of clients) client.response.end();
      const closed = new Promise((resolve) => server.close(resolve));
      server.closeAllConnections();
      await closed;
    },
  };
}
