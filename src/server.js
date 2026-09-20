import { createServer } from "node:http";
import { readFile } from "node:fs/promises";

const MAX_CLIENTS = 32;
const BLOCKED_CLIENT_MS = 15_000;

// A read-only local API. There is no control surface here at all: this process
// looks at the game, it never plays it.
export async function createStateServer(
  stream,
  { port = 8766, terrain = null } = {},
) {
  const assets = new Map(
    await Promise.all(
      [
        ["/", "index.html", "text/html; charset=utf-8"],
        ["/app.js", "app.js", "text/javascript; charset=utf-8"],
        ["/style.css", "style.css", "text/css; charset=utf-8"],
      ].map(async ([path, file, type]) => [
        path,
        {
          body: await readFile(new URL(`../public/${file}`, import.meta.url)),
          type,
        },
      ]),
    ),
  );
  const clients = new Set();
  let closing = false;
  let origin;

  // A sample that stopped arriving is worse than no sample: the dashboard would
  // keep drawing a worm that has not moved for a minute as if it were live.
  const current = () => {
    const state = stream.latest;
    if (
      state.status === "connected" &&
      Date.now() - Date.parse(state.capturedAt) >
        Math.max(2000, 3000 / stream.hz)
    ) {
      return {
        ...state,
        status: "stale",
        game: null,
        message: "No fresh browser sample is available.",
      };
    }
    return state;
  };
  const send = (client, state) => {
    if (client.blocked || client.response.destroyed) return;
    if (
      !client.response.write(
        `id: ${state.sessionId}:${state.sequence}\nevent: state\ndata: ${JSON.stringify(state)}\n\n`,
      )
    ) {
      // A consumer that cannot keep up skips frames rather than growing a
      // buffer; one that stays blocked is dropped by the heartbeat below.
      client.blocked = true;
      client.blockedAt = Date.now();
    }
  };
  const broadcast = (state) => {
    for (const client of clients) send(client, state);
  };
  stream.on("state", broadcast);

  const server = createServer(async (request, response) => {
    const json = (status, body) => {
      response.writeHead(status, {
        "Content-Type": "application/json; charset=utf-8",
      });
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
      return json(403, { error: "Use the local dashboard origin" });
    }
    if (request.method !== "GET") {
      response.setHeader("Allow", "GET");
      return json(405, { error: "Read-only API: GET is required" });
    }
    const path = new URL(request.url, origin).pathname;
    try {
      if (path === "/state") return json(200, current());
      if (path === "/health") {
        const state = current();
        return json(200, {
          ok: true,
          ready: state.status === "connected",
          status: state.status,
          sequence: state.sequence,
          terrain: {
            near: terrain?.near?.at ?? null,
            map: terrain?.map?.at ?? null,
          },
        });
      }
      // The worm's surroundings: the near window plus the measured facts about
      // the ground it is standing on.
      if (path === "/terrain") {
        if (!terrain) return json(404, { error: "Terrain sampling is off" });
        const patch = terrain.near ?? (await terrain.refresh("near"));
        return patch
          ? json(200, patch)
          : json(409, { error: "No terrain yet: join a room first" });
      }
      // Every pixel of the level, with the palette that colours it and the
      // material table that says what each one does.
      if (path === "/map") {
        if (!terrain) return json(404, { error: "Terrain sampling is off" });
        // A round change swaps the whole level. Waiting out the slow sampling
        // interval would answer with the map that is no longer being played.
        const playing = current().game?.map;
        const cached = terrain.map?.map;
        const level =
          cached &&
          (!playing ||
            (cached.name === playing.name &&
              cached.width === playing.width &&
              cached.height === playing.height))
            ? terrain.map
            : await terrain.refresh("map");
        return level
          ? json(200, level)
          : json(409, { error: "No terrain yet: join a room first" });
      }
      if (path === "/events") {
        if (clients.size >= MAX_CLIENTS)
          return json(503, { error: "Too many stream clients" });
        response.writeHead(200, {
          "Content-Type": "text/event-stream",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
        });
        response.write("retry: 1000\n\n");
        const client = { response, blocked: false, blockedAt: 0 };
        clients.add(client);
        response.on("close", () => clients.delete(client));
        response.on("error", () => clients.delete(client));
        response.on("drain", () => {
          client.blocked = false;
        });
        // A reconnecting consumer starts from the latest complete state; past
        // frames are never replayed.
        send(client, current());
        return;
      }
      const asset = assets.get(path);
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
  const heartbeat = setInterval(() => {
    for (const client of clients) {
      if (client.blocked) {
        if (Date.now() - client.blockedAt > BLOCKED_CLIENT_MS)
          client.response.destroy();
      } else if (current().status === "stale") send(client, current());
      else if (!client.response.write(": heartbeat\n\n")) {
        client.blocked = true;
        client.blockedAt = Date.now();
      }
    }
  }, 1000);
  heartbeat.unref();
  return {
    origin,
    async close() {
      closing = true;
      clearInterval(heartbeat);
      stream.off("state", broadcast);
      for (const client of clients) client.response.end();
      const closed = new Promise((resolve) => server.close(resolve));
      server.closeAllConnections();
      await closed;
    },
  };
}
