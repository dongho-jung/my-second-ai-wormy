// Chromium exposes window bounds over CDP. Placing the two windows is a
// convenience, so every step here is best effort: a failure costs a tidy layout
// and never the session.
const GAME_SHARE = 0.62;
const MIN_WIDTH = 480;

export function split(width, share = GAME_SHARE) {
  const game = Math.round(width * share);
  // Below roughly two minimum windows there is nothing useful to tile.
  if (width < MIN_WIDTH * 2) return null;
  return [game, width - game];
}

export async function setBounds(page, bounds) {
  const session = await page.context().newCDPSession(page);
  try {
    const { windowId } = await session.send("Browser.getWindowForTarget");
    await session.send("Browser.setWindowBounds", {
      windowId,
      bounds: { windowState: "normal", ...bounds },
    });
  } finally {
    await session.detach().catch(() => {});
  }
}

export async function tileWindows(gamePage, dashboardPage) {
  const screen = await gamePage
    .evaluate(() => [
      window.screen.availWidth,
      window.screen.availHeight,
      window.screen.availLeft ?? 0,
      window.screen.availTop ?? 0,
    ])
    .catch(() => null);
  if (!Array.isArray(screen) || screen.some((value) => !Number.isFinite(value)))
    return false;
  const [width, height, left, top] = screen;
  const columns = split(width);
  if (!columns) return false;
  const [gameWidth, dashboardWidth] = columns;
  try {
    await setBounds(gamePage, { left, top, width: gameWidth, height });
    await setBounds(dashboardPage, {
      left: left + gameWidth,
      top,
      width: dashboardWidth,
      height,
    });
    return true;
  } catch {
    return false;
  }
}
