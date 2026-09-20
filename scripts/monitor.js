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

  --port 8768   Local port; 0 picks an available one
  --open        Print the URL only (the default), never launch a browser`;

const { values } = parseArgs({
  options: {
    help: { type: "boolean", short: "h" },
    port: { type: "string", default: "8768" },
  },
});
if (values.help) {
  console.log(HELP);
  process.exit(0);
}

const server = await createMonitorServer({ port: Number(values.port) });
const runs = await listRuns();
console.log(`Training monitor: ${server.origin}  (Ctrl+C to stop)`);
console.log(
  runs.length
    ? `${runs.length} run(s) on disk, latest ${runs[0].id} (${runs[0].status})`
    : `no runs yet in ${server.dir.pathname} — start one with: npm run rollout`,
);
process.on("SIGINT", async () => {
  await server.close();
  process.exit(0);
});
