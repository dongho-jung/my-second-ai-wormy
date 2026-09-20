import { parseArgs } from "node:util";
import { readFile } from "node:fs/promises";
import { WebLieroObserver } from "./observer.js";
import { StateStream } from "./stream.js";
import { TerrainSampler } from "./terrain.js";
import { createStateServer } from "./server.js";
import { createLogger } from "./log.js";
import { setNickname } from "./room.js";
import { tileWindows } from "./windows.js";
import {
  DEFAULT_PROFILE,
  attach,
  findGamePages,
  launchDetached,
  launchOwned,
  openPage,
} from "./browser.js";

const HELP = `Wormy II — WebLiero state and terrain, read live and drawn

Usage: npm start -- [options]

npm start opens the game window and the dashboard side by side, then reads the
game 20 times a second. Join or create a room in the game window; a CAPTCHA, if
one is asked for, has to be completed there. Ctrl+C or closing the game window
stops.

  --url URL             Page to open (default: the WebLiero lobby)
  --port 8766           Local dashboard port; 0 picks an available one
  --browser-port 9334   Chromium debugging port for the game window
  --hz 20               State sampling rate, 1-60 Hz
  --terrain-ms 250      How often to read the worm's surroundings
  --map-ms 3000         How often to read the whole level
  --nickname NAME       Fill player setup automatically (1-25 characters)
  --browser-path PATH   Use an installed Chromium executable
  --profile PATH        Chromium profile to keep settings in
  --headless            Hide the game window
  --no-dashboard        Do not open the dashboard window automatically
  --own-browser         Close Chromium when this process exits
  --verbose             Log every terrain read failure too

By default Chromium is left running on its debugging port, and the next run
attaches to it — so the room, and the CAPTCHA already solved for it, survive a
restart. --own-browser turns that off.

Every launch reuses one profile, so nickname, key bindings and the rest of the
game's settings survive a browser restart too. Pass --profile to keep more than
one, or to put it somewhere else.

Read-only HTTP: GET /state, /events (SSE), /terrain, /map, /health.`;

function options() {
  const { values } = parseArgs({
    options: {
      help: { type: "boolean", short: "h" },
      url: { type: "string", default: "https://www.webliero.com/" },
      port: { type: "string", default: "8766" },
      "browser-port": { type: "string", default: "9334" },
      hz: { type: "string", default: "20" },
      "terrain-ms": { type: "string", default: "250" },
      "map-ms": { type: "string", default: "3000" },
      nickname: { type: "string" },
      "browser-path": { type: "string" },
      profile: { type: "string" },
      headless: { type: "boolean", default: false },
      dashboard: { type: "boolean", default: true },
      "own-browser": { type: "boolean", default: false },
      verbose: { type: "boolean", default: false },
    },
    allowNegative: true,
  });
  if (values.help) {
    console.log(HELP);
    return null;
  }
  const integer = (name, value, low, high) => {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < low || parsed > high)
      throw new Error(`--${name} must be an integer from ${low} to ${high}`);
    return parsed;
  };
  if (values.nickname !== undefined && !/^.{1,25}$/.test(values.nickname))
    throw new Error("--nickname must be 1 to 25 characters");
  const url = new URL(values.url);
  if (url.protocol !== "https:" || !/(^|\.)webliero\.com$/.test(url.hostname))
    throw new Error("--url must be an https URL on webliero.com");
  return {
    url: url.href,
    port: integer("port", values.port, 0, 65535),
    browserPort: integer("browser-port", values["browser-port"], 1024, 65535),
    hz: integer("hz", values.hz, 1, 60),
    nearMs: integer("terrain-ms", values["terrain-ms"], 50, 60_000),
    mapMs: integer("map-ms", values["map-ms"], 250, 600_000),
    nickname: values.nickname ?? (values.headless ? "Wormy observer" : null),
    browserPath: values["browser-path"],
    profilePath: values.profile,
    headless: values.headless,
    dashboard: values.dashboard && !values.headless,
    ownBrowser: values["own-browser"],
    verbose: values.verbose,
  };
}

async function warnOnRuntimeDrift(log) {
  const pinned = await readFile(new URL("../.nvmrc", import.meta.url), "utf8")
    .then((text) => text.trim())
    .catch(() => null);
  if (!pinned) return;
  const [wanted] = pinned.split(".");
  const [running] = process.version.replace(/^v/, "").split(".");
  if (wanted !== running)
    log.warn("runtime_drift", { running: process.version, pinned });
}

async function main() {
  const config = options();
  if (!config) return;
  const log = createLogger({ level: config.verbose ? "debug" : "info" });
  await warnOnRuntimeDrift(log);

  let browser;
  let keepBrowser = false;
  let observer;
  let stream;
  let terrain;
  let server;
  let stopping;
  let shutdownRequested = false;

  const stop = (error) => {
    shutdownRequested = true;
    if (stopping) return stopping;
    if (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
    stopping = (async () => {
      terrain?.stop();
      await stream?.stop().catch(() => {});
      await observer?.close().catch(() => {});
      // Only a browser this process owns belongs to it. Never send a close
      // through CDP: the attached room must survive a restart.
      if (!keepBrowser) await browser?.close().catch(() => {});
      await server?.close();
      log.info("stopped", { because: error ? error.message : "requested" });
      // Playwright has no public disconnect-only method for CDP. Once the loops
      // and the socket are drained, process exit releases the client connection
      // without closing the browser or any of its pages.
      if (keepBrowser) process.exit(process.exitCode ?? 0);
    })();
    return stopping;
  };

  let interrupts = 0;
  const signal = () => {
    shutdownRequested = true;
    // Ctrl+C has to end the process, always. A second one leaves immediately,
    // and even the first is backed by a deadline.
    if (++interrupts > 1) {
      console.error("Interrupted again: exiting now.");
      process.exit(130);
    }
    setTimeout(() => {
      console.error("Shutdown did not finish in time: exiting now.");
      process.exit(130);
    }, 15_000).unref();
    if (browser || stream?.running) void stop();
  };
  process.on("SIGINT", signal);
  process.on("SIGTERM", signal);

  try {
    // Three ways in, in order of preference: attach to the Chromium left by the
    // last run, start one that will outlive this one, or a plain owned browser.
    const owned = () =>
      launchOwned({
        headless: config.headless,
        executablePath: config.browserPath,
        profilePath: config.profilePath,
      });
    const opened = config.ownBrowser
      ? await owned()
      : await attach(config.browserPort, { timeoutMs: 1500 }).catch(
          async (error) => {
            log.debug("attach_unavailable", { message: error.message });
            return launchDetached({
              port: config.browserPort,
              executablePath: config.browserPath,
              headless: config.headless,
              profilePath: config.profilePath,
            }).catch(async (failure) => {
              log.warn("keep_browser_failed", { message: failure.message });
              return owned();
            });
          },
        );
    browser = opened.browser;
    keepBrowser = opened.detached;
    log.info("browser", {
      mode: opened.attached ? "attached" : keepBrowser ? "detached" : "owned",
      port: keepBrowser ? config.browserPort : null,
      profile: opened.attached
        ? null
        : (config.profilePath ?? DEFAULT_PROFILE),
    });
    if (shutdownRequested) return await stop();

    // Attaching reuses the window that is already in the room; anything else
    // needs a fresh page.
    const existing = opened.attached ? await findGamePages(browser) : [];
    const page = existing[0]?.page ?? (await openPage(browser));
    page.on("pageerror", (error) =>
      log.debug("page_error", { message: error.message }),
    );
    browser.on("disconnected", () => void stop());
    page.on("close", () => void stop());

    observer = new WebLieroObserver(page, { log: log.child("observer") });
    stream = new StateStream(observer, { hz: config.hz, log });
    terrain = new TerrainSampler(observer, {
      nearMs: config.nearMs,
      mapMs: config.mapMs,
      log,
    });
    server = await createStateServer(stream, { port: config.port, terrain });
    console.log(
      `Dashboard: ${server.origin}\nJSON: ${server.origin}/state · ${server.origin}/terrain · ${server.origin}/map`,
    );
    if (shutdownRequested) return await stop();

    if (!existing[0]?.inGame)
      await page.goto(config.url, {
        waitUntil: "domcontentloaded",
        timeout: 30_000,
      });
    if (shutdownRequested) return await stop();
    if (config.nickname) await setNickname(page, config.nickname);
    if (shutdownRequested) return await stop();

    if (config.dashboard) {
      // Its own window, so the game stays visible while you watch the readouts.
      // It shares the profile but not the game's storage: a different origin
      // never sees another one's.
      let dashboard = browser
        .contexts()
        .flatMap((context) => context.pages())
        .find((candidate) => candidate.url().startsWith(`${server.origin}/`));
      if (!dashboard) dashboard = await browser.contexts()[0].newPage();
      await dashboard.goto(server.origin, { timeout: 15_000 }).catch(() => {});
      const tiled = await tileWindows(page, dashboard);
      // The game window is the one being played in, so leave it focused.
      await page.bringToFront().catch(() => {});
      log.info("dashboard_window", { tiled });
    }
    if (shutdownRequested) return await stop();

    console.log(
      "Join or create a room in the game window to start collecting. Ctrl+C to stop.",
    );
    terrain.start();
    await stream.start();
  } catch (error) {
    if (shutdownRequested) await stop();
    else await stop(error);
  } finally {
    await stopping;
    process.off("SIGINT", signal);
    process.off("SIGTERM", signal);
  }
}

await main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
