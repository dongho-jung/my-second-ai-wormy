// Drives WebLiero's own UI. Every selector here was read off the live page;
// nothing patches or replaces game code.
import { createNullLogger } from "./log.js";

export const LOBBY_CREATE = 'button.btn-subtle[data-hook="create"]';
export const DIALOG_CREATE = 'button.btn-1[data-hook="create"]';
export const SPECTATING = '.game-view [data-hook="spectating"]';
export const TEAM_BUTTONS = {
  any: `${SPECTATING} [data-hook="join"]`,
  alpha: `${SPECTATING} [data-hook="join-t1"]`,
  bravo: `${SPECTATING} [data-hook="join-t2"]`,
};

export class RoomError extends Error {}

// The game stacks its panels inside one popup container, so a hit test can land
// on that ancestor even when the button itself is visible and enabled. The
// fallback fires a real click event on the game's own button.
export async function clickThrough(locator, { timeout = 8000 } = {}) {
  try {
    await locator.click({ timeout });
    return "click";
  } catch (error) {
    if (
      !/intercepts pointer events|not stable|outside of the viewport/.test(
        error.message,
      )
    )
      throw error;
    await locator.dispatchEvent("click");
    return "dispatch";
  }
}

// The Player Setup dialog appears a few seconds after the page boots, so a bare
// presence check races it and silently leaves the dialog covering the game.
export async function setNickname(page, nickname, { timeoutMs = 20_000, colour = null } = {}) {
  const field = page.getByPlaceholder("Nickname");
  try {
    await field.waitFor({ state: "visible", timeout: timeoutMs });
  } catch {
    // Already set up in this profile: the dialog never shows again.
    return false;
  }
  await field.fill(nickname);
  // Player Setup carries the worm's colour as three sliders beside the name.
  // Three worms in one identical shade are three worms nobody watching can tell
  // apart, including whoever is trying to work out which one just shot them.
  if (colour) {
    const [red, green, blue] = colour;
    for (const [hook, value] of [["rslider", red], ["gslider", green], ["bslider", blue]]) {
      const slider = page.locator(`[data-hook="${hook}"]`).first();
      if (!(await slider.count().catch(() => 0))) continue;
      // These are `type=range`, which Playwright's fill() refuses outright.
      // Setting the value and raising the events the widget listens for is
      // what dragging the handle does, without the arithmetic of where to drag.
      await slider
        .evaluate((el, wanted) => {
          el.value = String(wanted);
          el.dispatchEvent(new Event("input", { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
        }, value)
        .catch(() => {});
    }
  }
  // The live dialog carries a data-hook; fall back to the button's name so a
  // markup change costs a slower start rather than a failed one.
  const hooked = page.locator('[data-hook="ok"]');
  const confirm = (await hooked.count())
    ? hooked.first()
    : page.getByRole("button", { name: "Ok", exact: true });
  await clickThrough(confirm, { timeout: 5000 });
  return true;
}

/**
 * WebLiero gates room creation behind a CAPTCHA. That is an anti-automation
 * control on somebody else's service: it is reported and waited out here, never
 * worked around.
 */
export async function captchaVisible(page) {
  return page
    .evaluate(
      () =>
        !document.querySelector(".game-view") &&
        (Boolean(document.querySelector('iframe[src*="recaptcha"]')) ||
          document.body.innerText.includes("Only humans")),
    )
    .catch(() => false);
}

export async function inRoom(page) {
  return page
    .evaluate(() => Boolean(document.querySelector(".game-view")))
    .catch(() => false);
}

/** Make a room through the lobby, the way a player would. */
export async function createRoom(
  page,
  {
    name = "wormy",
    // Room size is not the number of worms being driven. It is private, so
    // there is no reason to make it exactly big enough and then have nowhere
    // for a person to sit down and play.
    maxPlayers = 20,
    isPublic = false,
    timeoutMs = 600_000,
    pollMs = 1000,
    onCaptcha = () => {},
    now = () => Date.now(),
    log = createNullLogger(),
  } = {},
) {
  await page.locator('[data-hook="rooms-btn"]').click({ timeout: 10_000 });
  await page.locator(LOBBY_CREATE).click({ timeout: 10_000 });
  await page.locator('input[data-hook="name"]').fill(name, { timeout: 10_000 });
  await page
    .locator('select[data-hook="max-pl"]')
    .selectOption(String(maxPlayers), { timeout: 10_000 });
  const publicBox = page.locator('input[data-hook="public"]');
  if ((await publicBox.isChecked()) !== isPublic) {
    // The checkbox itself sits under a styled span, so the label is what a
    // click can actually land on.
    await publicBox.locator("xpath=..").click({ timeout: 10_000 });
  }
  await page.locator(DIALOG_CREATE).click({ timeout: 10_000 });

  const deadline = now() + timeoutMs;
  let announced = false;
  let previous = null;
  while (now() < deadline) {
    const captcha = await captchaVisible(page);
    const entered = await inRoom(page);
    // Every change of phase is reported, so a wait is explained rather than
    // looking like a hang.
    const phase = `${entered ? "in-room" : "lobby"}/${captcha ? "captcha" : "clear"}`;
    if (phase !== previous) {
      log.info("create_room_phase", { phase });
      previous = phase;
    }
    if (captcha && !announced) {
      announced = true;
      onCaptcha();
    }
    if (entered && !captcha) return true;
    await page.waitForTimeout(pollMs);
  }
  throw new RoomError(
    announced
      ? "Still waiting on the CAPTCHA in the game window"
      : "Room creation did not complete",
  );
}

/**
 * The link other players join on. Creating a room does not change the address
 * bar, so it comes from the game's own Room Link dialog — the same menu entry a
 * player uses. Call it only once a team has been joined, or the spectating
 * popup covers the menu button.
 */
export async function roomUrl(page) {
  const url = page.url();
  if (/[?&]c=/.test(url)) return url;
  const link = page.locator('input[data-hook="link"]');
  if (!(await link.count())) {
    await clickThrough(page.locator('.game-view [data-hook="menu"]'));
    const menu = page.locator(".dropmenu-view");
    await menu.first().waitFor({ state: "visible", timeout: 8000 });
    const entry = menu.getByText("Room Link", { exact: true });
    if (!(await entry.count())) throw new RoomError("The game menu has no Room Link entry");
    await clickThrough(entry.first());
    await link.waitFor({ state: "visible", timeout: 8000 });
  }
  // The dialog is drawn before the link is written into it, so the first read
  // comes back empty.
  let value = "";
  for (let attempt = 0; attempt < 10 && !/[?&]c=/.test(value); attempt++) {
    if (attempt) await page.waitForTimeout(500);
    value = await link.inputValue({ timeout: 5000 }).catch(() => "");
  }
  await page
    .locator('.dialog [data-hook="close"]')
    .last()
    .click({ timeout: 5000 })
    .catch(() => {});
  if (/[?&]c=/.test(value)) return value;
  throw new RoomError("The Room Link dialog held no room id");
}

/** Going straight to the link is enough; Player Setup still comes first. */
export async function joinRoom(page, url, { nickname, timeoutMs = 45_000 } = {}) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
  if (nickname) await setNickname(page, nickname);
  await page.waitForSelector(".game-view", { timeout: timeoutMs });
}

/**
 * Take a seat. Team buttons only exist in team modes, so a deathmatch falls
 * back to the single Join Game button.
 */
export async function joinTeam(page, preference = "any", { timeoutMs = 30_000 } = {}) {
  await page.waitForSelector(SPECTATING, { timeout: timeoutMs });
  const order =
    preference === "any"
      ? ["any", "alpha", "bravo"]
      : [preference, "any", preference === "alpha" ? "bravo" : "alpha"];
  for (const choice of order) {
    const button = page.locator(TEAM_BUTTONS[choice]);
    if (!(await button.count())) continue;
    if (!(await button.first().isVisible())) continue;
    await clickThrough(button.first(), { timeout: 10_000 });
    // The key-layout greeting lands right after joining and would otherwise
    // keep every later key press blocked behind it.
    await dismissHowToPlay(page).catch(() => {});
    return choice;
  }
  throw new RoomError("No join button was available: the room may be full");
}

/**
 * WebLiero greets a joining player with a "How to play" dialog holding the
 * default key layout. It sits in the popup container and blocks key input, so
 * this ticks its own "don't show again" box and presses Ok — what a player
 * does. The checkbox hook is unique to that dialog, so nothing else is clicked.
 */
export async function dismissHowToPlay(page, { timeoutMs = 4000 } = {}) {
  const popups = page.locator('.game-view [data-hook="popups"]');
  const remember = popups.locator('input[data-hook="dont-show-again"]');
  if (!(await remember.count())) return false;
  if (!(await remember.isChecked()))
    await clickThrough(remember.locator("xpath=.."), { timeout: timeoutMs });
  const confirm = popups.getByRole("button", { name: "Ok", exact: true });
  if (await confirm.count()) await clickThrough(confirm.first(), { timeout: timeoutMs });
  return true;
}

/** True once the player owns a living worm, which is what driving requires. */
export function spawned(state) {
  if (state?.status !== "connected") return false;
  const self = state.game?.players?.find(
    (player) => player.id === state.game.localPlayerId,
  );
  return Boolean(self?.alive && self.worm);
}
