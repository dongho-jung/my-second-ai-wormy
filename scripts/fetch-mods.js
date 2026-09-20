// Community mods, downloaded beside the other artifacts.
//
// res.dat carries the six mods the client ships, and rooms run more than that.
// A policy has to be trained under whichever one the room is set to — the
// weapons and the constants both change — so the mod has to come from wherever
// it is published, not only from the file the client bundles.
//
//     npm run mods                 # every mod listed below
//     npm run mods -- cs_rewormed  # just the one
import { mkdir, writeFile } from "node:fs/promises";
import { MODS_DIR } from "../src/env/engine.js";

/** Where each mod is published. Keyed by the name `--mod` takes. */
const MODS = {
  cs_rewormed: {
    from: "https://gitlab.com/webliero/webliero-mods/-/raw/master/kami/cs_rewormed",
    files: ["mod.json5", "sprites.wlsprt", "readme.md"],
  },
};

const asked = process.argv.slice(2);
const wanted = asked.length ? asked : Object.keys(MODS);
for (const name of wanted) {
  const mod = MODS[name];
  if (!mod) {
    console.error(`no mod called ${name}: known are ${Object.keys(MODS).join(", ")}`);
    process.exitCode = 1;
    continue;
  }
  const into = new URL(`${name}/`, MODS_DIR);
  await mkdir(into, { recursive: true });
  for (const file of mod.files) {
    const response = await fetch(`${mod.from}/${file}`);
    if (!response.ok) {
      // readme is a courtesy; the other two are the mod.
      if (file === "readme.md") continue;
      throw new Error(`${name}/${file}: ${response.status} ${response.statusText}`);
    }
    await writeFile(new URL(file, into), Buffer.from(await response.arrayBuffer()));
  }
  console.log(`${name} -> ${new URL(into).pathname}`);
}
