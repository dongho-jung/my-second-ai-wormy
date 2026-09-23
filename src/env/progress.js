// Whether a worm is actually getting anywhere.
//
// Health and kills say nothing about a worm that has dug itself into a hole, or
// one that walks the same twenty pixels back and forth for a minute. Both look
// exactly like a worm that is fine and simply not fighting yet. These are the
// measurements that tell them apart, kept per worm and per episode.
//
// Two different failures, two different measurements, because one number cannot
// catch both. Standing still is a displacement across a window of time. Going in
// circles is covering ground and arriving nowhere, which only a memory of where
// it has already been can see.

export const PROGRESS_DEFAULTS = {
  // The grid the visit memory is kept on. A worm is about 7 px tall, so 16 px
  // is a stride or two: coarse enough that dodging in place is not "new ground".
  cellPx: 16,
  // How far back the displacement is measured. 60 decisions is four seconds of
  // game time at the default frameskip, and a worm that walks for four seconds
  // covers about a hundred pixels.
  window: 60,
  stuckPx: 14,
  // How long a cell stays remembered. Coming back after this long is a return,
  // not a circle.
  revisitMemory: 150,
  goalRadiusPx: 24,
  // A jump further than this in one step is a respawn, not walking.
  teleportPx: 60,
  // Existing movement runs paused a goal's deadline while a worm was dead.
  // A speed phase enables this so respawn time cannot look like fast travel.
  goalClockWhileDead: false,
};

export class Progress {
  constructor(options = {}) {
    this.options = { ...PROGRESS_DEFAULTS, ...options };
    this.trail = new Float64Array(this.options.window * 2);
    this.visited = new Map();
    this.reset();
  }

  /**
   * A destination, set now rather than at the episode boundary. The distance
   * restarts from wherever the worm is, so the first step after this is not
   * paid for the jump from the old goal to the new one.
   */
  setGoal(goal, position = null) {
    this.goal = goal ?? null;
    const initialDistance =
      this.goal && position
        ? Math.hypot(position.x - this.goal.x, position.y - this.goal.y)
        : null;
    // Keep the old shaping boundary: the first update establishes the distance
    // used by goalDelta and does not earn progress for changing destinations.
    this.goalDistance = null;
    // The best straight-line distance reached so far. Some routes have to go
    // away from the destination before they can go around a wall or climb a
    // ledge. `goalBestDelta` rewards only a new closest point: the detour is
    // neutral instead of punished, and walking back and forth cannot collect
    // the same progress twice.
    this.goalBestDistance = initialDistance;
    this.skipGoalBestCredit = false;
    // The distance that actually has to be covered. Arrival is a circle rather
    // than one exact pixel, so paying or reporting the whole centre-to-centre
    // distance would make a perfect straight route look inefficient by the
    // radius of that circle.
    this.goalDirectPx =
      initialDistance === null
        ? null
        : Math.max(0, initialDistance - this.options.goalRadiusPx);
    this.goalPathPx = 0;
    this.goalLastPosition = this.goal && position ? [position.x, position.y] : null;
    this.goalSteps = 0;
  }

  /** A new episode: forget the trail, the visits and the goal. */
  reset({ goal = null } = {}) {
    this.trail.fill(0);
    this.filled = 0;
    this.at = 0;
    this.visited.clear();
    this.step = 0;
    this.stuckSteps = 0;
    this.goal = goal;
    this.goalDistance = null;
    this.goalBestDistance = null;
    this.skipGoalBestCredit = false;
    this.goalDirectPx = null;
    this.goalPathPx = 0;
    this.goalLastPosition = null;
    // Decisions spent on the current goal, for a caller that gives up on one.
    this.goalSteps = 0;
    this.cell = -1;
    return this;
  }

  /** A worm that respawned is somewhere else entirely; none of the trail holds. */
  restart() {
    this.filled = 0;
    this.at = 0;
    this.stuckSteps = 0;
    this.cell = -1;
    this.goalDistance = null;
    // A respawn may put the worm closer to its goal. Remember the previous
    // record, but do not pay for a teleport on the first live frame.
    this.skipGoalBestCredit = true;
    // A death or a respawn must not look like a very fast piece of travel.
    // Keep the goal and its clock, but start measuring its physical path again
    // from the first live position after the jump.
    this.goalLastPosition = null;
  }

  /**
   * One decision's worth of movement. `alive` false holds everything where it
   * is: a worm waiting to respawn is not stuck, it is dead.
   */
  update(position, alive = true) {
    const { window, stuckPx, cellPx, revisitMemory, goalRadiusPx, teleportPx } =
      this.options;
    // In a speed phase, time spent dead is still time spent failing to reach
    // the destination. Older runs keep their paused clock unless they opt in.
    if (this.goal && (alive || this.options.goalClockWhileDead)) this.goalSteps++;
    if (!alive) {
      this.restart();
      return this.facts(0, false, 0, 0, 0, false);
    }
    this.step++;

    // A respawn moves a worm across the map between one step and the next, and
    // nothing before it says anything about where it is now.
    const previous = this.filled > 0 ? this.recent(0) : null;
    const teleported = Boolean(
      previous && Math.hypot(position.x - previous[0], position.y - previous[1]) > teleportPx,
    );
    if (teleported) {
      this.restart();
    }

    const oldest = this.filled >= window ? this.recent(window - 1) : null;
    const movedPx = oldest
      ? Math.hypot(position.x - oldest[0], position.y - oldest[1])
      : Infinity;
    const stuck = movedPx < stuckPx;
    this.stuckSteps = stuck ? this.stuckSteps + 1 : 0;

    this.push(position);

    // Where it is, on the coarse grid. Only crossing into another cell counts:
    // holding a position is what the displacement measurement is for.
    const cell =
      Math.floor(position.y / cellPx) * 0x10000 + Math.floor(position.x / cellPx);
    let novel = 0;
    let revisit = 0;
    if (cell !== this.cell) {
      const seen = this.visited.get(cell);
      if (seen === undefined) novel = 1;
      else if (this.step - seen <= revisitMemory) revisit = 1;
      this.visited.set(cell, this.step);
      this.cell = cell;
    }

    let goalDelta = 0;
    let goalBestDelta = 0;
    let reachedGoal = false;
    if (this.goal) {
      const distance = Math.hypot(position.x - this.goal.x, position.y - this.goal.y);
      if (this.goalDirectPx === null) {
        this.goalDirectPx = Math.max(0, distance - goalRadiusPx);
      }
      if (this.goalLastPosition) {
        const travelled = Math.hypot(
          position.x - this.goalLastPosition[0],
          position.y - this.goalLastPosition[1],
        );
        if (travelled <= teleportPx) this.goalPathPx += travelled;
      }
      this.goalLastPosition = [position.x, position.y];
      goalDelta = this.goalDistance === null ? 0 : this.goalDistance - distance;
      this.goalDistance = distance;
      if (this.goalBestDistance === null || this.skipGoalBestCredit || teleported) {
        this.goalBestDistance = Math.min(this.goalBestDistance ?? distance, distance);
        this.skipGoalBestCredit = false;
      } else if (distance < this.goalBestDistance) {
        goalBestDelta = this.goalBestDistance - distance;
        this.goalBestDistance = distance;
      }
      if (distance <= goalRadiusPx) {
        // Arriving is paid once. Leaving it set would pay a worm to sit on the
        // spot for the rest of the episode.
        reachedGoal = true;
        this.goal = null;
        this.goalDistance = null;
        this.goalLastPosition = null;
      }
    }
    return this.facts(movedPx, stuck, novel, revisit, goalDelta, reachedGoal, goalBestDelta);
  }

  /**
   * How much of this map one cell is.
   *
   * Covering ground was worth a flat amount per cell, which made it worth
   * eleven times more on a 2126x920 community map than on a stock 504x350 one
   * — and on the big maps wandering paid better than winning, so a run duly
   * stopped fighting. A share of the map means the same thing everywhere.
   */
  sized(level) {
    const { cellPx } = this.options;
    this.cells = Math.max(
      1,
      Math.floor(level.width / cellPx) * Math.floor(level.height / cellPx),
    );
  }

  facts(movedPx, stuck, novel, revisit, goalDelta, reachedGoal, goalBestDelta = 0) {
    return {
      cells: this.cells ?? 1,
      movedPx: Number.isFinite(movedPx) ? movedPx : null,
      stuck,
      stuckSteps: this.stuckSteps,
      novel,
      revisit,
      goalDelta,
      goalBestDelta,
      reachedGoal,
      goalDistance: this.goalDistance,
      goalSteps: this.goalSteps ?? 0,
      // These three are non-zero only on the decision that completes a goal.
      // The environment consumes them before assigning the next one.
      goalDirectPx: reachedGoal ? (this.goalDirectPx ?? 0) : 0,
      goalPathPx: reachedGoal ? this.goalPathPx : 0,
      goalSpeed: reachedGoal && this.goalSteps > 0
        ? (this.goalDirectPx ?? 0) / this.goalSteps
        : 0,
      cellsVisited: this.visited.size,
    };
  }

  /** The position `back` steps ago, as a two-element view of the ring. */
  recent(back) {
    const { window } = this.options;
    const index = (this.at - 1 - back + window * 2) % window;
    return [this.trail[index * 2], this.trail[index * 2 + 1]];
  }

  push(position) {
    this.trail[this.at * 2] = position.x;
    this.trail[this.at * 2 + 1] = position.y;
    this.at = (this.at + 1) % this.options.window;
    if (this.filled < this.options.window) this.filled++;
  }
}
