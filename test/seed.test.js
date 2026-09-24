import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { createMonitorServer } from "../src/train/monitor.js";

test("trainers receive the same immutable seed through the monitor, including after restart", async () => {
  const folder = await mkdtemp(join(tmpdir(), "wormy-seed-"));
  const dir = pathToFileURL(`${folder}/`);
  await mkdir(new URL("source/", dir));
  await writeFile(new URL("source/best.pt", dir), "original champion");
  let server;
  const start = () => createMonitorServer({ port: 0, dir, seedId: "trial", seedRun: "source", basePath: "/b" });
  try {
    server = await start();
    const read = async () => {
      const response = await fetch(`${server.origin}/b/seeds/trial.pt`);
      assert.equal(response.status, 200);
      const body = Buffer.from(await response.arrayBuffer());
      assert.equal(response.headers.get("X-Checkpoint-SHA256"), createHash("sha256").update(body).digest("hex"));
      return body.toString();
    };
    assert.deepEqual(await Promise.all([read(), read()]), ["original champion", "original champion"]);
    await writeFile(new URL("source/best.pt", dir), "later policy");
    assert.equal(await read(), "original champion");
    await server.close();
    server = await start();
    assert.equal(await read(), "original champion");
    assert.equal((await fetch(`${server.origin}/b/seeds/unknown.pt`)).status, 404);
  } finally {
    await server?.close();
    await rm(folder, { recursive: true, force: true });
  }
});
