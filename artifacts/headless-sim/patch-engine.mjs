// Regenerate engine.js from the shipped bundle: replace the trailing u.js()
// boot call with an export of the closure's classes, so Node can reach them
// without running the game's UI startup.
import { readFileSync, writeFileSync } from "node:fs";
const NAMES = ["Pa", "V", "Z", "ja", "Ia", "yc", "sc", "$c", "S", "ua", "ad", "eb", "bd",
  "t", "u", "fa", "H", "xa", "wb", "ub", "P", "A", "ib", "U", "Jc", "Qc", "xd", "ta", "N", "Ra", "zd", "v"];
const src = readFileSync("game-v20.orig.js", "utf8");
const at = src.lastIndexOf("u.js()");
if (at < 0) throw new Error("boot call u.js() not found — is this the v20 bundle?");
const exports = "var __e={};" + NAMES.map((n) => `try{__e["${n}"]=${n}}catch(_){}`).join("") + "da.__wl=__e;";
writeFileSync("engine.js", src.slice(0, at) + exports + src.slice(at + "u.js()".length));
console.log(`engine.js written: boot call at ${at} replaced with ${NAMES.length} class exports`);
