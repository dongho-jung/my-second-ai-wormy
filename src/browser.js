import { spawn } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

// One profile, kept between runs. A throwaway profile costs the WebLiero
// player setup, the key bindings and every other setting on every launch, which
// is a worse tax than the stale state a fresh profile avoids.
export const DEFAULT_PROFILE = join(homedir(), ".cache", "wormy-ii", "chrome-profile");

// Readies the profile a launch is about to use and returns its path.
export async function prepareProfile(path) {
  const profile = path ?? DEFAULT_PROFILE;
  await mkdir(profile, { recursive: true });
  await forgetLastSession(profile);
  return profile;
}

// A profile that has been used before comes back holding the tabs it had open:
// three runs later the window is five tabs deep, two of them dead dashboards on
// ports nobody is listening on any more. Measured on this profile — it happens
// after a clean exit too, so telling it the last exit was fine is not enough.
//
// The restore data lives in Default/Sessions and is only that: which tabs were
// open. Settings, cookies, local storage and the game's own saved keys are
// elsewhere in the profile and are never touched, which is the whole point of
// keeping one profile in the first place.
async function forgetLastSession(profile) {
  await rm(join(profile, "Default", "Sessions"), {
    recursive: true,
    force: true,
  }).catch(() => {});
  const path = join(profile, "Default", "Preferences");
  try {
    const preferences = JSON.parse(await readFile(path, "utf8"));
    await writeFile(
      path,
      JSON.stringify({
        ...preferences,
        profile: {
          ...preferences.profile,
          exit_type: "Normal",
          exited_cleanly: true,
        },
        // 5 is "open the new tab page", the setting a profile that restores
        // nothing would have.
        session: { ...preferences.session, restore_on_startup: 5, startup_urls: [] },
      }),
    );
  } catch {
    // A profile this run is creating has no preferences to correct yet.
  }
}

const LAUNCH_ARGS = [
  "--enable-unsafe-swiftshader",
  // The dashboard sits next to the game; without these Chromium throttles
  // timers and rAF in whichever window is not on top, stalling capture.
  "--disable-backgrounding-occluded-windows",
  "--disable-renderer-backgrounding",
  "--disable-background-timer-throttling",
];

// Playwright adds these when it launches a browser itself. Spawning Chromium
// directly skips them, and a fresh profile then opens the first-run intro page
// — which plays music over the game.
const FIRST_RUN_ARGS = [
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-fre",
  "--disable-search-engine-choice-screen",
  "--disable-features=ChromeWhatsNewUI,PrivacySandboxSettings4,Translate",
  "--disable-sync",
  "--propagate-iph-for-testing",
  "--disable-session-crashed-bubble",
  "--hide-crash-restore-bubble",
];

export class BrowserError extends Error {}

// A browser Playwright owns dies with this process, which means every restart
// costs a fresh room and another CAPTCHA. Spawning Chromium ourselves with a
// debugging port lets the next run attach to the same window: the game stays in
// the room while the code reading it is replaced.
export async function launchDetached({
  port,
  executablePath,
  headless = false,
  profilePath,
  timeoutMs = 30_000,
  spawnImpl = spawn,
  now = () => Date.now(),
}) {
  const profile = await prepareProfile(profilePath);
  const child = spawnImpl(
    executablePath ?? chromium.executablePath(),
    [
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      ...(headless ? ["--headless=new"] : []),
      ...FIRST_RUN_ARGS,
      ...LAUNCH_ARGS,
      "about:blank",
    ],
    { detached: true, stdio: "ignore" },
  );
  // Detaching means Chromium outlives this process on purpose.
  child.unref();
  try {
    const browser = await waitForEndpoint(port, { timeoutMs, now });
    return { browser, pid: child.pid, profile, port, detached: true };
  } catch (error) {
    // Never leave a window sitting at about:blank with nothing driving it.
    try {
      process.kill(child.pid);
    } catch {}
    throw error;
  }
}

export async function attach(port, { timeoutMs = 2000, now = () => Date.now() } = {}) {
  const browser = await waitForEndpoint(port, { timeoutMs, now }).catch(() => {
    throw new BrowserError(`No Chromium is listening on ${port}.`);
  });
  return { browser, port, detached: true, attached: true };
}

async function waitForEndpoint(port, { timeoutMs, now }) {
  const deadline = now() + timeoutMs;
  let lastError;
  while (now() < deadline) {
    try {
      // Short per-attempt timeout: a Chromium on its way out still completes
      // the handshake and then drops the socket, and waiting Playwright's full
      // default on that one attempt is what turned a quick relaunch into a
      // thirty second stall.
      const browser = await chromium.connectOverCDP(
        `http://127.0.0.1:${port}`,
        { timeout: 2000 },
      );
      if (browser.isConnected() && browser.contexts().length) return browser;
      // close() on a CDP connection disconnects; it never closes the browser.
      await browser.close().catch(() => {});
      lastError = new Error("that Chromium was shutting down");
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new BrowserError(
    `Could not reach Chromium on ${port}: ${lastError?.message ?? "timed out"}`,
  );
}

// Both an attached browser and one launched on a persistent profile hand back
// an existing context. Asking such a browser for a fresh context is what raises
// "Browser context management is not supported", so pages are always opened in
// the context that is already there.
//
// A blank tab is only free once: claiming them keeps two openPage calls from
// handing back the same about:blank.
const claimed = new WeakSet();

export async function openPage(browser) {
  const context = browser.contexts()[0];
  if (!context)
    throw new BrowserError("The attached Chromium exposes no browser context");
  const idle = context
    .pages()
    .find((page) => /^about:blank$/.test(page.url()) && !claimed.has(page));
  if (idle) {
    claimed.add(idle);
    return idle;
  }
  return context.newPage();
}

// Finds pages already sitting in a WebLiero game, so a restart can pick up the
// room that is still open instead of joining again.
export async function findGamePages(browser) {
  const pages = [];
  for (const context of browser.contexts()) {
    for (const page of context.pages()) {
      if (!/webliero\.com/.test(page.url())) continue;
      const inGame = await page
        .evaluate(() => Boolean(document.querySelector(".game-view")))
        .catch(() => false);
      pages.push({ page, inGame, url: page.url() });
    }
  }
  // Pages already in a room come first; a lobby tab is only a fallback.
  return pages.sort((a, b) => Number(b.inGame) - Number(a.inGame));
}

// A page of its own, in a window of its own. Playwright's newPage opens a tab,
// and two tabs of one window cannot be put side by side.
export async function openWindow(browser, url, { timeoutMs = 15_000 } = {}) {
  const context = browser.contexts()[0];
  if (!context)
    throw new BrowserError("The Chromium exposes no browser context");
  const before = new Set(context.pages());
  const cdp = await browser.newBrowserCDPSession();
  try {
    await cdp.send("Target.createTarget", { url, newWindow: true });
  } finally {
    await cdp.detach().catch(() => {});
  }
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const opened = context.pages().find((page) => !before.has(page));
    if (opened) return opened;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new BrowserError(`No window opened for ${url}`);
}

// Leftovers from earlier runs: restored tabs, dead dashboards, the blank page a
// launch starts on. Only the kinds of page this tool opens are closed, and only
// in its own profile — anything else a window holds is left where it is.
export async function tidy(browser, { keep = [], origin = null } = {}) {
  const kept = new Set(keep.filter(Boolean));
  const strays = [];
  for (const context of browser.contexts())
    for (const page of context.pages()) {
      if (kept.has(page) || page.isClosed()) continue;
      const url = page.url();
      if (
        /^(about:blank|chrome-error:|chrome:\/\/new-?tab)/.test(url) ||
        /^https?:\/\/(www\.)?webliero\.com([/?#]|$)/.test(url) ||
        (origin && url.startsWith(origin))
      )
        strays.push(page);
    }
  for (const page of strays) await page.close().catch(() => {});
  return strays.length;
}

// The same profile, in a browser that dies with this process.
export async function launchOwned({ headless, executablePath, profilePath }) {
  const profile = await prepareProfile(profilePath);
  try {
    const context = await chromium.launchPersistentContext(profile, {
      headless,
      executablePath,
      timeout: 20_000,
      viewport: null,
      args: [...(headless ? [] : FIRST_RUN_ARGS), ...LAUNCH_ARGS],
    });
    return { browser: context.browser(), detached: false };
  } catch (error) {
    // Chromium holds a lock on a profile it has open, and the message it gives
    // for that says nothing about profiles.
    throw new BrowserError(
      `Could not open the Chromium profile at ${profile}: ${error.message}\n` +
        "Another Chromium may already have it open — close that one, or pass --profile with a different path.",
    );
  }
}
