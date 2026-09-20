import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

// One profile, kept between runs. A throwaway profile costs the WebLiero
// player setup, the key bindings and every other setting on every launch, which
// is a worse tax than the stale state a fresh profile avoids.
export const DEFAULT_PROFILE = join(homedir(), ".cache", "wormy-ii", "chrome-profile");

async function profileDir(path) {
  const profile = path ?? DEFAULT_PROFILE;
  await mkdir(profile, { recursive: true });
  return profile;
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
  const profile = await profileDir(profilePath);
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
      return await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
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

// The same profile, in a browser that dies with this process.
export async function launchOwned({ headless, executablePath, profilePath }) {
  const profile = await profileDir(profilePath);
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
