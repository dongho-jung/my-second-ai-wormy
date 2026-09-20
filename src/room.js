// Drives WebLiero's own UI. Every selector here was read off the live page;
// nothing patches or replaces game code.

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
export async function setNickname(page, nickname, { timeoutMs = 20_000 } = {}) {
  const field = page.getByPlaceholder("Nickname");
  try {
    await field.waitFor({ state: "visible", timeout: timeoutMs });
  } catch {
    // Already set up in this profile: the dialog never shows again.
    return false;
  }
  await field.fill(nickname);
  // The live dialog carries a data-hook; fall back to the button's name so a
  // markup change costs a slower start rather than a failed one.
  const hooked = page.locator('[data-hook="ok"]');
  const confirm = (await hooked.count())
    ? hooked.first()
    : page.getByRole("button", { name: "Ok", exact: true });
  await clickThrough(confirm, { timeout: 5000 });
  return true;
}
