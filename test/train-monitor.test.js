import test from "node:test";
import assert from "node:assert/strict";
import { get } from "node:http";
import { mkdtemp, rm, writeFile, mkdir, readFile, rename } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createMonitorServer } from "../src/train/monitor.js";
import {
  createRun,
  describeRun,
  listRuns,
  newRunId,
  readRecords,
} from "../src/train/recorder.js";

async function runsDir() {
  const path = await mkdtemp(join(tmpdir(), "wormy-runs-"));
  return pathToFileURL(`${path}/`);
}

async function waitUntil(check) {
  const until = Date.now() + 3000;
  while (Date.now() < until) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 15));
  }
  assert.fail("viewer lifecycle did not settle");
}

async function viewerFixture(dir) {
  const script = new URL("viewer.cjs", dir);
  await writeFile(script, `
    const fs = require('node:fs');
    const http = require('node:http');
    const path = process.argv[process.argv.indexOf('--checkpoint') + 1];
    const data = JSON.parse(fs.readFileSync(path, 'utf8'));
    fs.appendFileSync(path + '.events', 'start ' + data.version + '\\n');
    process.on('SIGTERM', () => setTimeout(() => {
      fs.appendFileSync(path + '.events', 'stop ' + data.version + '\\n');
      process.exit(0);
    }, 30));
    const server = http.createServer((req, res) => {
      res.end(JSON.stringify(data));
      if (req.url === '/exit') setImmediate(() => process.exit(0));
    });
    server.listen(0, '127.0.0.1', () => {
      if (!data.pending) console.log('viewer http://127.0.0.1:' + server.address().port);
    });
  `);
  return fileURLToPath(script);
}

test("Watch reloads changed best.pt, reuses unchanged snapshots and releases the old viewer", { timeout: 10_000 }, async () => {
  const dir = await runsDir();
  const run = await createRun({ dir });
  const checkpoint = new URL(`${run.id}/best.pt`, dir);
  await writeFile(checkpoint, JSON.stringify({ version: 1 }));
  const server = await createMonitorServer({ port: 0, dir, viewerScript: await viewerFixture(dir) });
  const start = () => fetch(`${server.origin}/runs/${run.id}/watch`, { method: "POST" }).then((r) => r.json());
  try {
    const [first, duplicate] = await Promise.all([start(), start()]);
    assert.equal(duplicate.url, first.url);
    assert.equal(duplicate.alreadyRunning, true);
    assert.equal((await (await fetch(first.url)).json()).version, 1);
    const replacement = new URL(`${run.id}/next.pt`, dir);
    await writeFile(replacement, JSON.stringify({ version: 2 }));
    await rename(replacement, checkpoint);
    const second = await start();
    assert.equal((await (await fetch(second.url)).json()).version, 2);
    assert.match(await readFile(new URL(`${run.id}/best.pt.events`, dir), "utf8"), /start 1\nstop 1\nstart 2/);
    await fetch(`${second.url}/exit`);
    await waitUntil(async () => (await (await fetch(`${server.origin}/watch`)).json()).watching === null);
    const restarted = await start();
    assert.equal((await (await fetch(restarted.url)).json()).version, 2);
    await server.close();
    assert.match(await readFile(new URL(`${run.id}/best.pt.events`, dir), "utf8"), /stop 2\n$/);
  } finally {
    await server.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("stopping Watch cancels pending starts after client disconnect and permits a fresh start", { timeout: 10_000 }, async () => {
  const dir = await runsDir();
  const run = await createRun({ dir });
  const checkpoint = new URL(`${run.id}/best.pt`, dir);
  await writeFile(checkpoint, JSON.stringify({ version: 1, pending: true }));
  const server = await createMonitorServer({ port: 0, dir, viewerScript: await viewerFixture(dir) });
  const endpoint = `${server.origin}/runs/${run.id}/watch`;
  const abort = new AbortController();
  try {
    const disconnected = fetch(endpoint, { method: "POST", signal: abort.signal }).catch(() => null);
    await waitUntil(async () => (await (await fetch(`${server.origin}/watch`)).json()).watching === run.id);
    const queued = fetch(endpoint, { method: "POST" }).then((r) => r.json());
    abort.abort();
    await disconnected;
    await fetch(`${server.origin}/watch/stop`, { method: "POST" });
    assert.match((await queued).error, /cancelled|stopped/i);
    assert.equal((await (await fetch(`${server.origin}/watch`)).json()).watching, null);
    await writeFile(checkpoint, JSON.stringify({ version: 2 }));
    const fresh = await (await fetch(endpoint, { method: "POST" })).json();
    assert.equal((await (await fetch(fresh.url)).json()).version, 2);
  } finally {
    await server.close();
    await rm(dir, { recursive: true, force: true });
  }
});

/** Reads named SSE frames off a live response, one at a time. */
function frames(response) {
  const reader = response.body.getReader();
  let text = "";
  return {
    async next(name) {
      for (;;) {
        const match = text.match(
          new RegExp(`event: ${name}\\ndata: (.*)\\n\\n`),
        );
        if (match) {
          text = text.slice(match.index + match[0].length);
          return JSON.parse(match[1]);
        }
        const chunk = await reader.read();
        assert.equal(chunk.done, false, `stream ended before a ${name} frame`);
        text += new TextDecoder().decode(chunk.value);
      }
    },
    cancel: () => reader.cancel(),
  };
}

test("a run is a directory of records that can be read back as it grows", async () => {
  const dir = await runsDir();
  try {
    const run = await createRun({ dir, label: "one", meta: { seed: 5 } });
    await run.record({ step: 1, reward: 0.5 });
    await run.record({ step: 2, reward: 1.5 });
    const first = await readRecords(dir, run.id);
    assert.equal(first.records.length, 2);
    assert.equal(first.records[0].reward, 0.5);
    assert.ok(first.records[0].at, "every record is stamped when it was written");

    // A follower asks for what has appeared since the offset it was given.
    const none = await readRecords(dir, run.id, first.bytes);
    assert.deepEqual(none.records, []);
    assert.equal(none.bytes, first.bytes);
    await run.record({ step: 3, reward: 2.5 });
    const next = await readRecords(dir, run.id, first.bytes);
    assert.deepEqual(
      next.records.map((one) => one.step),
      [3],
    );

    const listed = await listRuns(dir);
    assert.equal(listed.length, 1);
    assert.equal(listed[0].status, "running");
    assert.equal(listed[0].label, "one");
    assert.equal(listed[0].meta.seed, 5);
    await run.close({ status: "done", episodes: 3 });
    const closed = await describeRun(dir, run.id);
    assert.equal(closed.status, "done");
    assert.equal(closed.meta.episodes, 3);
    assert.ok(closed.endedAt);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("a half-written line and a directory that is not a run are ignored", async () => {
  const dir = await runsDir();
  try {
    const run = await createRun({ dir });
    await run.record({ step: 1 });
    // A trainer killed mid-write leaves a torn line; everything before it counts.
    await writeFile(new URL(`${run.id}/metrics.jsonl`, dir), '{"step":2,"rew', {
      flag: "a",
    });
    const { records, bytes } = await readRecords(dir, run.id);
    assert.deepEqual(
      records.map((one) => one.step),
      [1],
    );
    await mkdir(new URL("not-a-run/", dir), { recursive: true });
    const listed = await listRuns(dir);
    assert.deepEqual(
      listed.map((one) => one.id),
      [run.id],
    );
    // The torn tail is not counted, so the next append is read from before it.
    assert.ok(bytes < listed[0].bytes);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("run ids sort by time and do not collide", () => {
  const first = newRunId(new Date("2026-09-20T10:00:00.000Z"));
  const later = newRunId(new Date("2026-09-20T10:00:01.000Z"));
  assert.ok(first < later, `${first} should sort before ${later}`);
  assert.notEqual(newRunId(), newRunId());
  assert.match(first, /^[\dA-Za-z-]+$/);
});

test("the monitor serves runs and follows a live one, read-only", { timeout: 10_000 }, async () => {
  const dir = await runsDir();
  const run = await createRun({ dir, label: "live", meta: { policy: "random" } });
  await run.record({ step: 1, reward: -1 });
  const server = await createMonitorServer({ port: 0, dir });
  try {
    const health = await (await fetch(`${server.origin}/health`)).json();
    assert.deepEqual(health, { ok: true, runs: 1, running: 1, clients: 0 });

    const page = await fetch(`${server.origin}/`);
    assert.equal(page.status, 200);
    assert.match(page.headers.get("content-type"), /text\/html/);
    assert.match(page.headers.get("content-security-policy"), /default-src 'self'/);

    const listed = await (await fetch(`${server.origin}/runs`)).json();
    assert.equal(listed.runs[0].id, run.id);
    const one = await (await fetch(`${server.origin}/runs/${run.id}`)).json();
    assert.equal(one.records.length, 1);
    assert.equal(one.run.meta.policy, "random");
    assert.equal(one.run.checkpoint, null, "a run with no saved policy cannot be watched");

    // A live run: connect, take what is already there, then watch it arrive.
    const stream = frames(await fetch(`${server.origin}/events`));
    assert.equal((await stream.next("runs")).runs.length, 1);
    assert.equal((await stream.next("run")).run.id, run.id);
    const opening = await stream.next("records");
    assert.equal(opening.reset, true);
    assert.deepEqual(
      opening.records.map((record) => record.step),
      [1],
    );
    await run.record({ step: 2, reward: 0.25 });
    const appended = await stream.next("records");
    assert.deepEqual(
      appended.records.map((record) => record.step),
      [2],
      "only what is new, never the whole file again",
    );
    await stream.cancel();

    // The only thing it will do besides read is start a viewer, and that only
    // on the one route.
    const post = await fetch(`${server.origin}/runs`, { method: "POST" });
    assert.equal(post.status, 405);
    assert.equal(post.headers.get("allow"), "GET, POST");
    const remove = await fetch(`${server.origin}/runs/${run.id}/watch`, {
      method: "DELETE",
    });
    assert.equal(remove.status, 405);
    assert.equal((await fetch(`${server.origin}/nope`)).status, 404);
    assert.equal((await fetch(`${server.origin}/runs/missing`)).status, 404);
    // A run id becomes a path segment, so it is refused rather than cleaned up.
    const escape = await fetch(`${server.origin}/runs/${encodeURIComponent("../../etc")}`);
    assert.equal(escape.status, 400);
    const crossOrigin = await fetch(`${server.origin}/runs`, {
      headers: { Origin: "http://evil.test" },
    });
    assert.equal(crossOrigin.status, 403);
    // fetch refuses to set Host, so this one goes out through node:http.
    const spoofed = await new Promise((resolve) =>
      get(`${server.origin}/runs`, { headers: { Host: "evil.test" } }, resolve),
    );
    spoofed.resume();
    assert.equal(spoofed.statusCode, 403);
  } finally {
    await run.close({ status: "stopped" });
    await server.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("the monitor answers with nothing at all when no run has been made", async () => {
  const dir = await runsDir();
  await rm(dir, { recursive: true, force: true });
  const server = await createMonitorServer({ port: 0, dir });
  try {
    const listed = await (await fetch(`${server.origin}/runs`)).json();
    assert.deepEqual(listed.runs, []);
    const stream = frames(await fetch(`${server.origin}/events`));
    assert.deepEqual((await stream.next("runs")).runs, []);
    await stream.cancel();
  } finally {
    await server.close();
  }
});
