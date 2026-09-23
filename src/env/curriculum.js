// How far a destination is drawn, decided by whether the worm has been getting
// there.
//
// The first curriculum grew the radius with the clock: 96 px at the start,
// 1,600 px three fifths of the way through, whatever the policy had learned.
// Measured, that outran it — at 130 px the worms reached six destinations a
// match, and by the time the radius had moved out to 250 px they reached
// three, with the goals moving away faster than the walking and the jumping
// and the rope were being learned. Nothing in the schedule could notice.
//
// So this one reads the outcomes instead. Every destination ends one of two
// ways, reached or given up on, and over a window of them the share that were
// reached says whether the current distance is easy or hard. Easy moves the
// limit out a notch; hard brings it back a notch; in between it stays. The
// notches are small and the window is long enough that a lucky match does not
// move it, and it never leaves the range it was given.
export const CURRICULUM_DEFAULTS = {
  // Destinations per decision: reached and given up on together.
  window: 30,
  // Share reached at or above which the radius grows.
  up: 0.85,
  // Share reached at or below which it shrinks.
  down: 0.5,
  // How much, each time: a tenth either way.
  step: 0.1,
};

const clamp = (value, low, high) => (value < low ? low : value > high ? high : value);

export class GoalCurriculum {
  constructor({ from, to, start = null, window, up, down, step } = {}) {
    if (!(from > 0) || !(to >= from)) {
      throw new Error(`a goal curriculum needs 0 < from <= to, got ${from}..${to}`);
    }
    this.from = from;
    this.to = to;
    this.window = window ?? CURRICULUM_DEFAULTS.window;
    this.up = up ?? CURRICULUM_DEFAULTS.up;
    this.down = down ?? CURRICULUM_DEFAULTS.down;
    this.step = step ?? CURRICULUM_DEFAULTS.step;
    // A run carrying on from a checkpoint starts where the radius had got to.
    this.radius = clamp(start ?? from, from, to);
    this.reached = 0;
    this.missed = 0;
    // Decisions taken so far, for whoever is charting them.
    this.decisions = 0;
  }

  /** One destination's outcome. Returns the radius to draw the next one at. */
  record(reached) {
    if (reached) this.reached++;
    else this.missed++;
    const seen = this.reached + this.missed;
    if (seen < this.window) return this.radius;
    const share = this.reached / seen;
    if (share >= this.up) this.radius = Math.min(this.to, this.radius * (1 + this.step));
    else if (share <= this.down) this.radius = Math.max(this.from, this.radius / (1 + this.step));
    this.reached = 0;
    this.missed = 0;
    this.decisions++;
    return this.radius;
  }
}

/**
 * The same outcome feedback, applied to time instead of distance.
 *
 * A movement policy first learns to reach destinations with a generous
 * deadline. Once that is reliable, repeatedly shortening a fixed schedule by
 * hand just guesses at what it can do. This tightens the deadline one notch
 * when the current one is easy, loosens it when too many goals are missed, and
 * stays within the requested range. Values are decisions, not engine ticks.
 */
export class GoalDeadlineCurriculum {
  constructor({ from, to, start = null, window, up, down, step } = {}) {
    if (!(to > 0) || !(from >= to)) {
      throw new Error(`a goal deadline curriculum needs from >= to > 0, got ${from}..${to}`);
    }
    this.from = from;
    this.to = to;
    this.window = window ?? CURRICULUM_DEFAULTS.window;
    this.up = up ?? CURRICULUM_DEFAULTS.up;
    this.down = down ?? CURRICULUM_DEFAULTS.down;
    this.step = step ?? CURRICULUM_DEFAULTS.step;
    this.patience = clamp(start ?? from, to, from);
    this.reached = 0;
    this.missed = 0;
    this.decisions = 0;
  }

  /** One destination's outcome. Returns the deadline for the next one. */
  record(reached) {
    if (reached) this.reached++;
    else this.missed++;
    const seen = this.reached + this.missed;
    if (seen < this.window) return this.patience;
    const share = this.reached / seen;
    if (share >= this.up) {
      this.patience = Math.max(this.to, this.patience / (1 + this.step));
    } else if (share <= this.down) {
      this.patience = Math.min(this.from, this.patience * (1 + this.step));
    }
    this.reached = 0;
    this.missed = 0;
    this.decisions++;
    return this.patience;
  }
}
