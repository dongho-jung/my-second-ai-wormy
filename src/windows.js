// Chromium exposes window bounds over CDP. Placing the two windows is a
// convenience, so every step here is best effort: a failure costs a tidy layout
// and never the session.
const GAME_SHARE = 0.5;
const MIN_WIDTH = 480;

export function split(width, share = GAME_SHARE) {
  const game = Math.round(width * share);
  // Below roughly two minimum windows there is nothing useful to tile.
  if (width < MIN_WIDTH * 2) return null;
  return [game, width - game];
}

async function windowFor(page) {
  const session = await page.context().newCDPSession(page);
  try {
    const { windowId } = await session.send("Browser.getWindowForTarget");
    return { session, windowId };
  } catch (error) {
    await session.detach().catch(() => {});
    throw error;
  }
}

export async function setBounds(page, bounds) {
  const { session, windowId } = await windowFor(page);
  try {
    await session.send("Browser.setWindowBounds", {
      windowId,
      bounds: { windowState: "normal", ...bounds },
    });
  } finally {
    await session.detach().catch(() => {});
  }
}

// Two halves of the usable screen, the game on the left. Returns false rather
// than guessing when the screen is too small to split, or when the two pages
// turn out to be tabs of one window — setting bounds twice on the same window
// only moves it twice, which is what made the layout look random.
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
  let game;
  let dashboard;
  try {
    game = await windowFor(gamePage);
    dashboard = await windowFor(dashboardPage);
    if (game.windowId === dashboard.windowId) return false;
    await game.session.send("Browser.setWindowBounds", {
      windowId: game.windowId,
      bounds: { windowState: "normal", left, top, width: gameWidth, height },
    });
    await dashboard.session.send("Browser.setWindowBounds", {
      windowId: dashboard.windowId,
      bounds: {
        windowState: "normal",
        left: left + gameWidth,
        top,
        width: dashboardWidth,
        height,
      },
    });
    return true;
  } catch {
    return false;
  } finally {
    await game?.session.detach().catch(() => {});
    await dashboard?.session.detach().catch(() => {});
  }
}
