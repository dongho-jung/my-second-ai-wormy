import { spawn } from "node:child_process";
import { stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { CHECKPOINTS } from "./recorder.js";

// The monitor owns one viewer. A checkpoint is a snapshot: reuse its process
// only while that exact file is still current, and release its port before
// starting a replacement. Pending starts belong to the same lifecycle.
export function createViewer({ dir, script, args = [] }) {
  let current = null;
  let queue = Promise.resolve();
  let generation = 0;

  async function stopEntry(entry) {
    if (!entry) return;
    entry.settle({ error: "Viewer stopped" });
    if (entry.child.exitCode === null && entry.child.signalCode === null) {
      entry.child.kill("SIGTERM");
      const timer = setTimeout(() => entry.child.kill("SIGKILL"), 10_000);
      timer.unref();
      await entry.exited;
      clearTimeout(timer);
    }
    if (current === entry) current = null;
  }

  async function start(id, epoch) {
    let checkpoint;
    let version;
    for (const name of CHECKPOINTS) {
      const path = new URL(`${id}/${name}`, dir);
      try {
        const info = await stat(path, { bigint: true });
        checkpoint = fileURLToPath(path);
        version = `${checkpoint}:${info.ino}:${info.size}:${info.mtimeNs}`;
        break;
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    if (!checkpoint) return { error: "This run has no saved checkpoint" };
    if (epoch !== generation) return { error: "Viewer start cancelled" };
    if (current?.version === version && current.child.exitCode === null
        && current.child.signalCode === null) {
      return { ...await current.ready, alreadyRunning: true };
    }
    await stopEntry(current);
    if (epoch !== generation) return { error: "Viewer start cancelled" };
    const child = spawn(process.execPath, [script, "--checkpoint", checkpoint, ...args], {
      cwd: fileURLToPath(new URL("../../", import.meta.url)),
      stdio: ["ignore", "pipe", "pipe"],
    });
    const entry = { id, version, child, reply: { url: null, run: id } };
    entry.exited = new Promise((resolve) => {
      child.once("close", resolve);
      child.once("error", resolve);
    });
    entry.ready = new Promise((resolve) => { entry.settle = resolve; });
    current = entry;
    let said = "";
    const listen = (chunk) => {
      said = (said + chunk).slice(-8192);
      const found = said.match(/viewer (https?:\/\/\S+)/);
      if (found) {
        entry.reply = { url: found[1], run: id };
        entry.settle(entry.reply);
      }
    };
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", listen);
    child.stderr.on("data", listen);
    child.on("error", (error) => entry.settle({ error: error.message }));
    child.on("close", (code) => {
      entry.settle({ error: said.trim().split("\n").slice(-4).join(" ") || `Viewer exited with ${code}` });
      if (current === entry) current = null;
    });
    const timer = setTimeout(() => entry.settle({ error: "The viewer did not start" }), 20_000);
    timer.unref();
    const result = await entry.ready;
    clearTimeout(timer);
    if (result.error) await stopEntry(entry);
    return result;
  }

  return {
    get current() { return current; },
    start(id) {
      const epoch = generation;
      const result = queue.then(() => start(id, epoch)).catch((error) => ({ error: error.message }));
      queue = result.then(() => {});
      return result;
    },
    async stop() {
      generation++;
      await stopEntry(current);
      await queue;
    },
  };
}
