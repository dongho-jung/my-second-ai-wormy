// Independent copies of one fixed movement scenario, drawn on top of each
// other as ghosts.
//
// Putting several worms in one WormEnv is not a fair race: they collide, see
// one another, share terrain damage and can pull or shoot each other. Here each
// racer owns a complete world. The only shared things are the scenario and the
// clock, so one racer cannot change another racer's path in any way.
import { WormEnv } from "./env.js";

const close = (left, right) => Math.abs(left - right) <= 1e-3;

function point(raw, name) {
  if (!Array.isArray(raw) || raw.length !== 2 || !raw.every(Number.isFinite)) {
    throw new Error(`race scenario ${name} must be a finite [x, y] point`);
  }
  return { x: Number(raw[0]), y: Number(raw[1]) };
}

function normaliseScenario(raw, index) {
  if (!raw || !Number.isInteger(raw.seed) || typeof raw.map !== "string") {
    throw new Error(`race scenario ${index + 1} needs a seed and map`);
  }
  return {
    ...raw,
    seed: raw.seed >>> 0,
    start: point(raw.start, `${index + 1} start`),
    goal: point(raw.goal, `${index + 1} goal`),
    detour: Boolean(raw.detour),
    dig: Boolean(raw.dig),
  };
}

/** A batch of policy attempts that share no mutable simulation state. */
export class GhostRace {
  constructor(engine, {
    racers,
    scenarios,
    levels,
    world = {},
    episodeTicks = null,
    // Once arrivals stop changing, the remaining ghosts are stragglers rather
    // than useful race information. Three game seconds still lets a close
    // pack finish without making every route wait for one failed sample.
    settleTicks = 180,
  }) {
    if (!Number.isInteger(racers) || racers < 1) {
      throw new Error(`ghost race racers must be a positive integer, got ${racers}`);
    }
    if (!Array.isArray(scenarios) || !scenarios.length) {
      throw new Error("ghost race needs at least one fixed scenario");
    }
    if (!Number.isInteger(settleTicks) || settleTicks < 0) {
      throw new Error(`ghost race settleTicks must be a non-negative integer, got ${settleTicks}`);
    }
    this.engine = engine;
    this.agents = racers;
    this.scenarios = scenarios.map(normaliseScenario);
    this.levels = levels;
    this.index = 0;
    this.current = this.scenarios[0];
    this.episode = 0;
    this.finished = [];
    this.done = true;
    this.settleTicks = settleTicks;
    this.lastFinishTick = null;
    this.settled = false;

    const level = () => {
      const found = this.levels.get(this.current.map)
        ?? this.levels.get(this.current.map.toLowerCase());
      if (!found) {
        throw new Error(
          `fixed race map ${this.current.map} is not installed; refusing to show another map`,
        );
      }
      return found;
    };
    const starts = () => this.current.start;
    const goals = () => ({ ...this.current.goal, detour: this.current.detour, dig: this.current.dig });
    const options = {
      ...world,
      agents: 1,
      stagger: false,
      level,
      starts,
      goals,
      goalsPerEpisode: 1,
      endOnGoals: false,
      goalPatience: 0,
      ...(episodeTicks ? { episodeTicks } : {}),
    };
    this.envs = Array.from({ length: racers }, () => new WormEnv(engine, options));
    this.spec = this.envs[0].spec;
    this.frameskip = this.envs[0].frameskip;
    this.episodeTicks = this.envs[0].episodeTicks;
    this.reset({ first: true });
  }

  get world() {
    return this.envs[0].world;
  }

  get observations() {
    return this.envs.map((env) => env.observations[0]);
  }

  get episodeSeed() {
    return this.current.seed;
  }

  get elapsedTicks() {
    return this.envs[0].info().elapsedTicks;
  }

  /** Start the next manifest entry, or the first one during construction. */
  reset({ first = false } = {}) {
    if (!first) this.index = (this.index + 1) % this.scenarios.length;
    this.current = this.scenarios[this.index];
    this.episode++;
    this.finished = Array.from({ length: this.agents }, () => null);
    this.done = false;
    this.lastFinishTick = null;
    this.settled = false;
    for (const env of this.envs) {
      env.reset({ seed: this.current.seed });
      const worm = env.worms[0];
      const goal = env.progress[0].goal;
      if (
        !close(worm.x, this.current.start.x)
        || !close(worm.y, this.current.start.y)
        || !goal
        || !close(goal.x, this.current.goal.x)
        || !close(goal.y, this.current.goal.y)
      ) {
        throw new Error("ghost race worlds did not receive the manifest's exact endpoints");
      }
    }
    return { observations: this.observations, info: this.info() };
  }

  /** One independently simulated action per ghost. */
  step(actions) {
    if (this.done) throw new Error("the ghost race is over: call reset() first");
    if (actions.length !== this.agents) {
      throw new Error(`expected ${this.agents} ghost actions, got ${actions.length}`);
    }
    const events = [];
    const rewards = [];
    const respawned = [];
    const restarted = [];
    let clockEnded = true;
    let newFinish = false;
    for (let racer = 0; racer < this.agents; racer++) {
      const env = this.envs[racer];
      const out = env.step([actions[racer]]);
      events.push(out.info.events[0]);
      rewards.push(out.rewards[0]);
      respawned.push(Boolean(out.respawned?.[0]));
      restarted.push(Boolean((out.restarted ?? out.respawned)?.[0]));
      clockEnded = clockEnded && out.done;
      if (!this.finished[racer] && (env.totals[0].goalsReached ?? 0) > 0) {
        const worm = env.worms[0];
        const decisions = env.totals[0].goalStepsReached ?? env.progress[0].goalSteps;
        this.finished[racer] = {
          seconds: decisions * env.frameskip / 60,
          x: worm.x,
          y: worm.y,
          pathPx: env.totals[0].goalPathPx ?? 0,
        };
        newFinish = true;
      }
    }
    if (newFinish) this.lastFinishTick = this.elapsedTicks;
    const allFinished = this.finished.every(Boolean);
    this.settled = Boolean(
      !allFinished
      && this.lastFinishTick !== null
      && this.elapsedTicks - this.lastFinishTick >= this.settleTicks
    );
    this.done = clockEnded || allFinished || this.settled;
    return {
      observations: this.observations,
      rewards,
      done: this.done,
      terminated: allFinished,
      respawned,
      restarted,
      info: { ...this.info(), events },
    };
  }

  /** The public facts the viewer needs, with no mutable engine objects. */
  info() {
    const order = this.finished
      .map((finish, id) => finish && ({ id, seconds: finish.seconds }))
      .filter(Boolean)
      .sort((a, b) => a.seconds - b.seconds || a.id - b.id);
    const ranks = new Map(order.map((one, index) => [one.id, index + 1]));
    return {
      elapsedTicks: this.elapsedTicks,
      episode: this.episode,
      seed: this.current.seed,
      scenario: this.index + 1,
      scenarios: this.scenarios.length,
      start: this.current.start,
      goal: this.current.goal,
      detour: this.current.detour,
      done: this.done,
      settled: this.settled,
      racers: this.envs.map((env, id) => {
        const worm = env.worms[0];
        const finish = this.finished[id];
        return {
          id,
          worm,
          loadout: env.loadouts[0],
          progress: env.progress[0],
          finish,
          rank: ranks.get(id) ?? null,
          dnf: this.done && !finish,
        };
      }),
    };
  }
}
