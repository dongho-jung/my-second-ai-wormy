import { createNullLogger } from "./log.js";

// Terrain is read on its own cadence, not with every state sample. Two
// pictures, for two different jobs:
//
//   near — the 426x240 window the game draws around the worm, at 2 px cells.
//          Cheap, so it is refreshed often enough to follow a worm digging.
//   map  — every pixel of the level with its palette and material table.
//          Expensive (a few hundred KB), so it is refreshed slowly; the shape
//          of a map changes far more slowly than a worm moves through it.
//
// Both go through the observer's own queue, so they never overlap a state
// sample. A read that fails between rounds leaves the last good picture up.
export class TerrainSampler {
  constructor(
    observer,
    {
      nearMs = 250,
      mapMs = 3000,
      log = createNullLogger(),
      now = () => Date.now(),
    } = {},
  ) {
    this.observer = observer;
    this.nearMs = nearMs;
    this.mapMs = mapMs;
    this.log = log;
    this.now = now;
    this.near = null;
    this.map = null;
    this.timers = [];
    this.busy = { near: false, map: false };
  }

  async refresh(kind) {
    if (this.busy[kind] || this.stopped) return this[kind];
    this.busy[kind] = true;
    try {
      const read = await this.observer.read(
        kind === "map" ? { terrain: true } : { terrainPatch: true },
      );
      if (read.status !== "connected") return this[kind];
      const game = read.game;
      if (kind === "map") {
        this.map = { at: this.now(), ...game };
      } else {
        // A dead or spectating player has no surroundings. Say so rather than
        // leaving the last living worm's window on screen.
        this.near = {
          at: this.now(),
          tick: game.tick,
          localPlayerId: game.localPlayerId,
          near: null,
          ...game.patch,
        };
      }
      return this[kind];
    } catch (error) {
      this.log.debug("terrain_read_failed", {
        kind,
        message: error.message,
      });
      return this[kind];
    } finally {
      this.busy[kind] = false;
    }
  }

  start() {
    if (this.timers.length) return this;
    for (const [kind, interval] of [
      ["near", this.nearMs],
      ["map", this.mapMs],
    ]) {
      const timer = setInterval(() => void this.refresh(kind), interval);
      timer.unref();
      this.timers.push(timer);
    }
    return this;
  }

  stop() {
    this.stopped = true;
    for (const timer of this.timers) clearInterval(timer);
    this.timers = [];
  }
}
