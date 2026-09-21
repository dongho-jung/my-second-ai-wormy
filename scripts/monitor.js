// The training page, on a port of its own.
//
// Read-only and independent of everything else: it needs no game, no browser and
// no running trainer. Leave it open while a run goes, or open it afterwards to
// look at what a finished run did.
import { parseArgs } from "node:util";
import { createMonitorServer } from "../src/train/monitor.js";
import { listRuns } from "../src/train/recorder.js";

const HELP = `Wormy II — training monitor

Usage: npm run monitor -- [options]

Serves a page that draws the runs under artifacts/runs/, following a running one
live. It only reads them.

  --port 8768   Port to serve on; 0 picks an available one
  --open        Print the URL only (the default), never launch a browser

Serving it somewhere other than this machine:

  --host            What to bind. 127.0.0.1 by default, which is a window onto
                    your own machine and nothing else. 0.0.0.0 in a container.
  --public-origin   Where a browser actually reaches it, when that is not where
                    it bound — https://dashboard.example.com , say. The page
                    refuses a request whose Host is neither, which is what
                    stops another site driving it through a browser that can.
  --base-path       A path it is mounted under, such as /ai-worm. The pages ask
                    for their own files by relative path, so one prefix moves
                    the whole thing and the viewer with it.

Each has an environment variable of its own — WORMY_HOST, WORMY_PUBLIC_ORIGIN,
WORMY_BASE_PATH — so a container needs no arguments at all.`;

const { values } = parseArgs({
  options: {
    help: { type: "boolean", short: "h" },
    port: { type: "string", default: "8768" },
    host: { type: "string", default: process.env.WORMY_HOST ?? "127.0.0.1" },
    "public-origin": { type: "string", default: process.env.WORMY_PUBLIC_ORIGIN ?? "" },
    "base-path": { type: "string", default: process.env.WORMY_BASE_PATH ?? "" },
  },
});
if (values.help) {
  console.log(HELP);
  process.exit(0);
}

const publicOrigin = values["public-origin"] || null;
const basePath = (values["base-path"] || "").replace(/\/+$/, "");
const server = await createMonitorServer({
  port: Number(values.port),
  host: values.host,
  publicOrigin,
  basePath,
  // The viewer the Watch button starts lives on its own port and so needs its
  // own address out there. Same hostname, one path further along: whoever
  // routes /ai-worm to this also routes /ai-worm/watch to the viewer.
  viewerHost: values.host,
  viewerOrigin: publicOrigin,
  viewerBasePath: basePath ? `${basePath}/watch` : "",
});
const runs = await listRuns();
console.log(
  `Training monitor: ${publicOrigin ? `${publicOrigin}${basePath}/` : server.origin}` +
    `  (Ctrl+C to stop)`,
);
console.log(
  runs.length
    ? `${runs.length} run(s) on disk, latest ${runs[0].id} (${runs[0].status})`
    : `no runs yet in ${server.dir.pathname} — start one with: npm run rollout`,
);
process.on("SIGINT", async () => {
  await server.close();
  process.exit(0);
});
