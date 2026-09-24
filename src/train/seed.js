import { createHash, randomUUID } from "node:crypto";
import { readFile, mkdir, writeFile, link, unlink } from "node:fs/promises";

// Pin once, before the trainer creates another run. Restarts keep the same
// bytes even when the source run's best.pt advances. Readers never see a copy
// in progress, and another monitor cannot overwrite an existing experiment.
export async function pinSeed(dir, id, runId) {
  if (!/^[A-Za-z0-9_-]+$/.test(id) || !/^[A-Za-z0-9_.-]+$/.test(runId) || runId === "." || runId === "..") {
    throw new Error("Invalid seed experiment or source run");
  }
  const folder = new URL("seeds/", dir);
  const target = new URL(`${id}.pt`, folder);
  await mkdir(folder, { recursive: true });
  let bytes;
  try {
    bytes = await readFile(target);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const source = await readFile(new URL(`${runId}/best.pt`, dir));
    const temporary = new URL(`.${id}-${process.pid}-${randomUUID()}.tmp`, folder);
    try {
      await writeFile(temporary, source, { flag: "wx" });
      try { await link(temporary, target); }
      catch (error) { if (error.code !== "EEXIST") throw error; }
    } finally {
      await unlink(temporary).catch((error) => { if (error.code !== "ENOENT") throw error; });
    }
    bytes = await readFile(target);
  }
  return { id, bytes, sha256: createHash("sha256").update(bytes).digest("hex") };
}
