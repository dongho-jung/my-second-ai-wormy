// The four files the headless engine is, downloaded from the game's own site.
//
// `artifacts/` is gitignored, so a fresh clone has no engine and every
// environment test skips with a reason. The script that used to fetch these
// lived inside `artifacts/` itself, which meant the one thing a new checkout
// needed was the one thing it did not have. This is that script, in the
// repository.
//
//     npm run engine
//
// Nothing here is redistributed: it is fetched from
// https://www.webliero.com/v/20/ at the version this project is locked to, and
// the bundle's SHA-256 is checked against the one `src/adapter-v20.js` carries.
// A mismatch means the site has moved to another build, and the right answer is
// to read the new one rather than to trust the old field mapping — so it fails.
import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { CLIENT_SHA256 } from "../src/adapter-v20.js";
import { ASSETS, DEFAULT_ENGINE_DIR } from "../src/env/engine.js";

const BASE = "https://www.webliero.com/v/20/";

/** Where each local name is served from. Two of them sit under `vendor/`. */
const REMOTE = {
  [ASSETS.bundle]: "game-min.js",
  [ASSETS.resources]: "res.dat",
  [ASSETS.json5]: "vendor/json5.min.js",
  [ASSETS.wasm]: "vendor/wasm-flate.wasm",
};

const into = new URL(process.argv[2] ? `${process.argv[2]}/` : DEFAULT_ENGINE_DIR, `file://${process.cwd()}/`);
await mkdir(into, { recursive: true });

for (const [name, path] of Object.entries(REMOTE)) {
  const from = new URL(path, BASE);
  const response = await fetch(from);
  if (!response.ok) {
    throw new Error(`${from} answered ${response.status}; the site may have moved to another version`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (name === ASSETS.bundle && sha256 !== CLIENT_SHA256) {
    throw new Error(
      `${path} is ${sha256}, not the ${CLIENT_SHA256} this project reads.\n` +
        "The field mapping in src/adapter-v20.js was read off that exact build. " +
        "Check the new one against it rather than moving the checksum.",
    );
  }
  await writeFile(new URL(name, into), bytes);
  console.log(`${name.padEnd(18)} ${bytes.length.toLocaleString().padStart(9)} bytes  ${sha256.slice(0, 12)}`);
}
console.log(`\nengine in ${fileURLToPath(into)}`);
