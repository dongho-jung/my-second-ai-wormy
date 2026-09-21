// Where a training run writes down how it is going.
//
// One directory per run, two files in it: `run.json` for the things that were
// decided once, and `metrics.jsonl` for one line per measurement. Append-only
// text on purpose — a run that crashes leaves everything it had already said,
// and the monitor can follow the file by reading the bytes that appeared since
// it last looked, without the trainer having to know anyone is watching.
//
// The records are deliberately free-form: anything numeric becomes a chart, so
// a trainer that starts logging a new quantity needs no change here or in the
// page.
import { mkdir, appendFile, access, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { randomBytes } from "node:crypto";

// Under artifacts/, which is gitignored: a run is a measurement of this machine,
// not a fact about the project.
export const DEFAULT_RUNS_DIR = new URL("../../artifacts/runs/", import.meta.url);

export const RUN_FILE = "run.json";
export const METRICS_FILE = "metrics.jsonl";
const SCHEMA_VERSION = 1;

/** A run id that sorts by time and cannot collide with a parallel worker's. */
export function newRunId(now = new Date()) {
  const stamp = now.toISOString().replace(/[:.]/g, "-").replace("Z", "");
  return `${stamp}-${randomBytes(2).toString("hex")}`;
}

const runDir = (dir, id) => new URL(`${id}/`, dir);

/**
 * Start writing a run. `meta` is whatever the run wants remembered about how it
 * was set up: seeds, environment options, the engine checksum.
 */
export async function createRun({
  dir = DEFAULT_RUNS_DIR,
  id = newRunId(),
  label = null,
  meta = {},
} = {}) {
  const path = runDir(dir, id);
  await mkdir(path, { recursive: true });
  const run = {
    schemaVersion: SCHEMA_VERSION,
    id,
    label,
    startedAt: new Date().toISOString(),
    endedAt: null,
    status: "running",
    meta,
  };
  const describe = () => writeFile(new URL(RUN_FILE, path), `${JSON.stringify(run, null, 2)}\n`);
  await describe();
  // Records are written in the order they were handed over, even when two
  // arrive in the same tick.
  let queue = Promise.resolve();
  const append = (line) => {
    queue = queue.then(() => appendFile(new URL(METRICS_FILE, path), line));
    return queue;
  };
  let records = 0;
  return {
    id,
    path,
    get records() {
      return records;
    },
    /** One measurement. `step` is the x axis the page plots everything against. */
    record(fields) {
      records++;
      return append(`${JSON.stringify({ at: new Date().toISOString(), ...fields })}\n`);
    },
    /** Something worth saying in words, shown on the page as it happened. */
    note(text, fields = {}) {
      return this.record({ ...fields, note: text });
    },
    async close({ status = "done", ...fields } = {}) {
      run.status = status;
      run.endedAt = new Date().toISOString();
      Object.assign(run.meta, fields);
      await queue;
      await describe();
      return run;
    },
  };
}

/** Every run on disk, newest first. */
export async function listRuns(dir = DEFAULT_RUNS_DIR) {
  let names;
  try {
    names = await readdir(dir);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const runs = await Promise.all(names.map((name) => describeRun(dir, name)));
  return runs
    .filter(Boolean)
    .sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

/** Checkpoints a run has saved, best first. A run with none cannot be watched. */
export const CHECKPOINTS = ["best.pt", "policy.pt"];

/**
 * How long a run may go without writing before it is taken to have stopped.
 *
 * A run's own file says "running" from the moment it is created and is only
 * rewritten by a clean exit, so anything that ends a trainer without letting
 * it unwind — a SIGKILL, an out-of-memory kill, a node going away — leaves a
 * file claiming to still be training. The honest answer is when it last
 * wrote. An update takes about two minutes at the defaults, so this is more
 * than two missed updates: long enough never to call a live run dead.
 */
export const SILENT_FOR_STOPPED_MS = 5 * 60 * 1000;

/** A run's own description, plus how much it has written so far. */
export async function describeRun(dir, id) {
  const path = runDir(dir, id);
  try {
    const run = JSON.parse(await readFile(new URL(RUN_FILE, path), "utf8"));
    const metrics = await stat(new URL(METRICS_FILE, path)).catch(() => null);
    let checkpoint = null;
    for (const name of CHECKPOINTS) {
      try {
        await access(new URL(name, path));
        checkpoint = name;
        break;
      } catch {
        // Not saved yet, or not saved at all: a rollout has no policy to keep.
      }
    }
    const updatedAt = (metrics?.mtime ?? new Date(run.startedAt)).toISOString();
    const silent = Date.now() - new Date(updatedAt).getTime() > SILENT_FOR_STOPPED_MS;
    return {
      ...run,
      // Only what it wrote can say it is running; the flag alone cannot.
      status: run.status === "running" && silent ? "stopped" : run.status,
      checkpoint,
      bytes: metrics?.size ?? 0,
      updatedAt,
    };
  } catch {
    // A directory that is not a run, or one caught mid-write: skip it rather
    // than fail the whole listing.
    return null;
  }
}

/**
 * The records a run has written from byte `from` onwards, and the offset to ask
 * from next time. Only whole lines are parsed: the trainer may be halfway
 * through writing the next one.
 */
export async function readRecords(dir, id, from = 0) {
  const file = new URL(`${id}/${METRICS_FILE}`, dir);
  let text;
  try {
    text = await readFile(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return { records: [], bytes: 0 };
    throw error;
  }
  const bytes = Buffer.byteLength(text);
  if (from >= bytes) return { records: [], bytes };
  const tail = from > 0 ? text.slice(byteToCharOffset(text, from)) : text;
  const complete = tail.lastIndexOf("\n");
  if (complete < 0) return { records: [], bytes: from };
  const lines = tail.slice(0, complete).split("\n").filter(Boolean);
  const records = [];
  for (const line of lines) {
    try {
      records.push(JSON.parse(line));
    } catch {
      // A torn line from a killed writer: everything before it still counts.
    }
  }
  return { records, bytes: from + Buffer.byteLength(tail.slice(0, complete + 1)) };
}

// The offsets on the wire are byte offsets, because that is what a file's size
// is; records are ASCII JSON in practice but a label can be anything.
function byteToCharOffset(text, byteOffset) {
  let bytes = 0;
  for (let index = 0; index < text.length; index++) {
    if (bytes >= byteOffset) return index;
    bytes += Buffer.byteLength(text[index]);
  }
  return text.length;
}
