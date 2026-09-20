import test from "node:test";
import assert from "node:assert/strict";
import { get } from "node:http";
import { StateStream } from "../src/stream.js";
import { TerrainSampler } from "../src/terrain.js";
import { createStateServer } from "../src/server.js";

// Reads the first SSE frame off a live response.
async function event(reader) {
  let text = "";
  while (!text.includes("\ndata: ")) {
    const chunk = await reader.read();
    assert.equal(chunk.done, false);
    text += new TextDecoder().decode(chunk.value);
  }
  return JSON.parse(text.split("\ndata: ")[1].split("\n")[0]);
}

function fakeObserver() {
  const reads = [];
  return {
    reads,
    read: async (options = {}) => {
      reads.push(options);
      if (options.terrain)
        return {
          status: "connected",
          game: { tick: 9, map: { name: "Fixture", width: 4, height: 2 }, data: "AAECAwQF" },
        };
      if (options.terrainPatch)
        return {
          status: "connected",
          game: {
            tick: 8,
            localPlayerId: 12,
            patch: {
            self: { x: 1, y: 2 },
            near: { origin: [0, 1], size: [1, 1], bounds: [4, 2], data: "AA==" },
          },
          },
        };
      return { status: "connected", game: { tick: 5 } };
    },
  };
}

test("HTTP snapshots, SSE, terrain and map are served read-only", { timeout: 10_000 }, async () => {
  const observer = fakeObserver();
  const stream = new StateStream(observer);
  const terrain = new TerrainSampler(observer, { now: () => 1000 });
  const server = await createStateServer(stream, { port: 0, terrain });
  try {
    const events = await fetch(`${server.origin}/events`);
    const reader = events.body.getReader();
    assert.equal((await event(reader)).status, "loading", "a new client starts from the latest state");
    stream.publish({ status: "connected", game: { tick: 1 } });
    assert.equal((await event(reader)).game.tick, 1);
    await reader.cancel();

    stream.publish({ status: "connected", game: { tick: 2 } });
    assert.equal((await (await fetch(`${server.origin}/state`)).json()).game.tick, 2);
    const health = await (await fetch(`${server.origin}/health`)).json();
    assert.equal(health.ready, true);

    // Nothing has been sampled yet, so the first request reads on demand.
    const patch = await (await fetch(`${server.origin}/terrain`)).json();
    assert.deepEqual(patch.self, { x: 1, y: 2 });
    assert.equal(patch.tick, 8);
    assert.equal(patch.at, 1000);
    const map = await (await fetch(`${server.origin}/map`)).json();
    assert.equal(map.map.name, "Fixture");
    assert.deepEqual(
      observer.reads.filter((options) => options.terrain || options.terrainPatch),
      [{ terrainPatch: true }, { terrain: true }],
      "terrain is read on its own options, never bundled into a state sample",
    );

    assert.equal((await fetch(`${server.origin}/nope`)).status, 404);
    const posted = await fetch(`${server.origin}/state`, { method: "POST" });
    assert.equal(posted.status, 405, "this API never accepts a write");
  } finally {
    terrain.stop();
    await server.close();
  }
});

test("a state sample that stopped arriving is reported as stale, not as live", async () => {
  const stream = new StateStream({ read: async () => ({}) }, { hz: 20 });
  const server = await createStateServer(stream, { port: 0 });
  try {
    stream.publish({ status: "connected", game: { tick: 3 } });
    stream.latest.capturedAt = new Date(Date.now() - 5000).toISOString();
    const state = await (await fetch(`${server.origin}/state`)).json();
    assert.equal(state.status, "stale");
    assert.equal(state.game, null);
    assert.equal((await (await fetch(`${server.origin}/health`)).json()).ready, false);
  } finally {
    await server.close();
  }
});

test("the local API refuses another origin and another host", async () => {
  const stream = new StateStream({ read: async () => ({}) });
  const server = await createStateServer(stream, { port: 0 });
  try {
    assert.equal(
      (await fetch(`${server.origin}/state`, { headers: { Origin: "http://evil.test" } })).status,
      403,
    );
    // fetch refuses to set Host, so this one goes out through node:http.
    const spoofed = await new Promise((resolve) =>
      get(`${server.origin}/state`, { headers: { Host: "evil.test" } }, resolve),
    );
    spoofed.resume();
    assert.equal(spoofed.statusCode, 403);
  } finally {
    await server.close();
  }
});

test("terrain endpoints say so when nothing is sampling", async () => {
  const stream = new StateStream({ read: async () => ({}) });
  const server = await createStateServer(stream, { port: 0 });
  try {
    assert.equal((await fetch(`${server.origin}/terrain`)).status, 404);
    assert.equal((await fetch(`${server.origin}/map`)).status, 404);
  } finally {
    await server.close();
  }
});

test("terrain reads that fail leave the last good picture up", async () => {
  let fail = false;
  const observer = {
    read: async () =>
      fail
        ? { status: "awaiting_room", game: null }
        : {
            status: "connected",
            game: { tick: 1, localPlayerId: 3, patch: { near: { data: "AA==" } } },
          },
  };
  const terrain = new TerrainSampler(observer, { now: () => 42 });
  const first = await terrain.refresh("near");
  assert.equal(first.tick, 1);
  fail = true;
  assert.equal((await terrain.refresh("near")).tick, 1, "a failed read never blanks the picture");
});
