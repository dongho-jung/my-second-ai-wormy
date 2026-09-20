import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, access } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { prepareProfile, tidy } from "../src/browser.js";
import { split } from "../src/windows.js";

test("the screen is split in half, and not at all when it is too narrow", () => {
  assert.deepEqual(split(1920), [960, 960]);
  assert.deepEqual(split(1512), [756, 756]);
  assert.equal(split(800), null);
});

test("preparing a profile drops the restored tabs and keeps the settings", async () => {
  const profile = await mkdtemp(join(tmpdir(), "wormy-profile-"));
  const preferences = join(profile, "Default", "Preferences");
  const sessions = join(profile, "Default", "Sessions");
  await mkdir(sessions, { recursive: true });
  await writeFile(join(sessions, "Session_1"), "tabs from the last run");
  await writeFile(
    preferences,
    JSON.stringify({
      profile: { exit_type: "Crashed", name: "Person 1" },
      // Everything a player set up in the game window lives in preferences
      // like these; losing them is the reason one profile is kept at all.
      browser: { window_placement: { left: 10 } },
    }),
  );
  assert.equal(await prepareProfile(profile), profile);
  await assert.rejects(access(sessions), "the tab restore data is gone");
  const after = JSON.parse(await readFile(preferences, "utf8"));
  assert.equal(after.profile.exit_type, "Normal");
  assert.equal(after.profile.exited_cleanly, true);
  assert.equal(after.session.restore_on_startup, 5);
  assert.equal(after.profile.name, "Person 1", "settings are left alone");
  assert.deepEqual(after.browser, { window_placement: { left: 10 } });
});

test("a profile that does not exist yet is created, not refused", async () => {
  const profile = join(await mkdtemp(join(tmpdir(), "wormy-profile-")), "new");
  assert.equal(await prepareProfile(profile), profile);
  await access(profile);
});

test("tidying closes this tool's leftovers and nothing else", async () => {
  const closed = [];
  const page = (url) => ({
    url: () => url,
    isClosed: () => false,
    close: async () => closed.push(url),
  });
  const keep = [page("https://www.webliero.com/?c=room"), page("http://127.0.0.1:8766/")];
  const pages = [
    ...keep,
    page("about:blank"),
    page("about:blank#probe"),
    page("chrome-error://chromewebdata/"),
    page("chrome://new-tab-page/"),
    page("https://www.webliero.com/"),
    page("http://127.0.0.1:8766/state"),
    page("https://github.com/dongho-jung"),
    page("https://webliero.com.evil.test/"),
  ];
  const browser = { contexts: () => [{ pages: () => pages }] };
  assert.equal(
    await tidy(browser, { keep, origin: "http://127.0.0.1:8766" }),
    6,
  );
  assert.deepEqual(closed, [
    "about:blank",
    "about:blank#probe",
    "chrome-error://chromewebdata/",
    "chrome://new-tab-page/",
    "https://www.webliero.com/",
    "http://127.0.0.1:8766/state",
  ]);
});
