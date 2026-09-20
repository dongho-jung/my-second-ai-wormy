// Pressing keys in a live game, the way a person would.
//
// The policy decides in bitmasks. A browser only has a keyboard, so this holds
// down the keys that produce the bitmask the policy asked for and lets go of
// the ones it did not — a difference per decision, not a fresh press every
// time. That matters for more than tidiness: jump and dig only fire on the
// press, so a key that is released and pressed again every decision would jump
// fifteen times a second, and one that is simply held jumps once. Held is what
// the training environment does, so held is what happens here.
//
// It never touches the game's input code. Bindings are read out of the game's
// own storage and the keys go in as real events.
import { createNullLogger } from "../log.js";
import { BINDINGS, codesByAction, keysForBits, missingBindings, pressesForAction } from "./keys.js";

/**
 * How long a tapped key stays down. The game samples the keyboard once per
 * 60 Hz frame, so a press has to outlast a frame to have happened at all, and
 * it has to be over before the next decision so the one after is a fresh press.
 */
const TAP_MS = 60;

export class ControlsError extends Error {}

export class Controls {
  constructor(page, { tapMs = TAP_MS, log = createNullLogger() } = {}) {
    this.page = page;
    this.tapMs = tapMs;
    this.log = log;
    this.held = new Set();
    this.tapping = new Map();
    this.codes = codesByAction(BINDINGS);
  }

  /** The bindings the game itself is using, not the ones we hope it has. */
  async readBindings() {
    return this.page
      .evaluate((fallback) => {
        const stored = localStorage.getItem("player_keys");
        return stored ? JSON.parse(stored) : fallback;
      }, BINDINGS)
      .catch(() => BINDINGS);
  }

  /**
   * Check the game agrees about which key is which before driving it. A policy
   * pressing D expecting to fire, in a game where D is bound to something else,
   * fails in a way that looks like the policy being bad.
   */
  async verify() {
    const bindings = await this.readBindings();
    const missing = missingBindings(bindings);
    if (missing.length) {
      throw new ControlsError(
        `these actions have no key in the game's settings: ${missing.join(", ")}. ` +
          "Set them in Settings > Input.",
      );
    }
    this.codes = codesByAction(bindings);
    this.log.info("bindings", { codes: this.codes });
    return this.codes;
  }

  /**
   * Hand one decision to the keyboard. `keys` is the engine's bitmask; the rope
   * and the weapon are messages in the game, so they are taps here.
   */
  async apply({ keys = 0, rope = 0, weapon = 0 } = {}) {
    const wanted = new Set();
    for (const action of keysForBits(keys)) {
      const code = this.codes[action];
      if (code) wanted.add(code);
    }
    for (const code of [...this.held]) {
      if (!wanted.has(code)) await this.up(code);
    }
    for (const code of wanted) {
      if (!this.held.has(code)) await this.down(code);
    }
    for (const action of pressesForAction({ rope, weapon })) {
      const code = this.codes[action];
      if (code) await this.tap(code);
    }
  }

  async down(code) {
    if (this.held.has(code)) return;
    this.held.add(code);
    await this.page.keyboard.down(code).catch((error) => {
      this.held.delete(code);
      throw error;
    });
  }

  async up(code) {
    if (!this.held.has(code)) return;
    this.held.delete(code);
    await this.page.keyboard.up(code).catch(() => {});
  }

  /** Down now, up shortly: the game has to see both edges. */
  async tap(code) {
    if (this.tapping.has(code)) return;
    await this.down(code);
    this.tapping.set(
      code,
      setTimeout(() => {
        this.tapping.delete(code);
        void this.up(code);
      }, this.tapMs),
    );
  }

  /** Let go of everything. A worm with a key stuck down walks into a wall. */
  async release() {
    for (const timer of this.tapping.values()) clearTimeout(timer);
    this.tapping.clear();
    for (const code of [...this.held]) await this.up(code);
  }
}
