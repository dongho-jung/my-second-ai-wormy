// Community map pools, downloaded beside the other artifacts.
//
// `npm run levels` fetches the game's own .lev pool. Rooms also run maps from
// the community repository, which publishes them as 8-bit palette PNGs — the
// same one-byte-per-pixel terrain in a different wrapper.
//
//     npm run maps                 # every pool below
//     npm run maps -- dsds-cs      # just the one
import { mkdir, writeFile } from "node:fs/promises";
import { MAPS_DIR } from "../src/env/engine.js";

const PROJECT = "webliero%2Fwebliero-maps";
const RAW = "https://gitlab.com/webliero/webliero-maps/-/raw/master";

/** Where each pool lives, and which of its files belong to it. */
const POOLS = {
  "dsds-cs": { path: "dsds", keep: (name) => name.startsWith("cs_") && name.endsWith(".png") },
};

const asked = process.argv.slice(2);
const wanted = asked.length ? asked : Object.keys(POOLS);
for (const pool of wanted) {
  const spec = POOLS[pool];
  if (!spec) {
    console.error(`no pool called ${pool}: known are ${Object.keys(POOLS).join(", ")}`);
    process.exitCode = 1;
    continue;
  }
  const listing = await fetch(
    `https://gitlab.com/api/v4/projects/${PROJECT}/repository/tree` +
      `?path=${encodeURIComponent(spec.path)}&ref=master&per_page=500`,
  );
  if (!listing.ok) throw new Error(`listing ${spec.path}: ${listing.status} ${listing.statusText}`);
  const names = (await listing.json())
    .filter((entry) => entry.type === "blob" && spec.keep(entry.name))
    .map((entry) => entry.name);
  const into = new URL(`${pool}/`, MAPS_DIR);
  await mkdir(into, { recursive: true });
  let written = 0;
  for (const name of names) {
    const response = await fetch(`${RAW}/${encodeURIComponent(spec.path)}/${encodeURIComponent(name)}`);
    if (!response.ok) {
      console.error(`${name}: ${response.status} ${response.statusText}`);
      continue;
    }
    await writeFile(new URL(encodeURIComponent(name), into), Buffer.from(await response.arrayBuffer()));
    written++;
  }
  console.log(`${pool}: ${written} maps -> ${new URL(into).pathname}`);
}
