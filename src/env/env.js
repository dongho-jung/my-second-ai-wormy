// The environment: worlds in, actions in, observations and rewards out.
//
// One instance is one match at a time — by default three worms in a free-for-all,
// which is a different game from a duel: the worm shooting at you is often not
// the one you are shooting at, and the third one is deciding which of you is
// worth interrupting. Everything that could make two runs of one seed differ
// goes through a single seeded generator, so a rollout is reproducible and a
// policy's bad episode can be replayed exactly.
import { applyNormalizedAction, normalizeAction } from "./actions.js";
import { makeRng, respawnWorm, watchDamage } from "./engine.js";
import { OBSERVATIONS, observationSpec, observe } from "./observation.js";
import { Progress } from "./progress.js";
import {
  addEvents,
  combatReward,
  DEFAULT_WEIGHTS,
  emptyEvents,
  tallyDamage,
} from "./reward.js";
import { viewFromWorld } from "./view.js";

// Stock Liero weapons that give a worm something to do at every range: shotgun,
// rifle, bazooka, mine, crackler. Pass "random" instead to draw five of the
// mod's forty per worm per episode, which is what teaches the game rather than
// the shotgun.
export const DEFAULT_LOADOUT = [0, 2, 3, 5, 10];

export const DEFAULTS = {
  // Three is a free-for-all, which is the interesting case; two is a duel and
  // more is a brawl. Nothing here is written for three in particular — the
  // observation, the attribution and the rewards are all built from this number.
  agents: 3,
  // How many other worms the vector describes. By default every one of them.
  // Pin it higher than `agents` and the same policy can play a duel and a
  // five-way without being retrained: the empty slots read as zeros.
  observationFoes: null,
  // Four ticks is a policy deciding at 15 Hz: about as often as a person
  // changes their mind, and four times less network to run.
  frameskip: 4,
  // A minute of game time. One core runs it in well under a second.
  episodeTicks: 3600,
  respawn: true,
  terminateOnKill: false,
  // Ticks between a policy choosing and the worm acting. The headless engine
  // has none; a live game has a network and a 60 Hz keyboard sampler. An agent
  // that has only ever played at zero latency learns timing that does not
  // survive the move, so train across a range and it stops mattering.
  inputLatencyTicks: 0,
  loadout: "random",
  // Both pictures by default. The terrain patch is about ten times the cost of
  // the vector, so a task that does not need it — walking somewhere, a first
  // check that the pipeline learns anything at all — should pass ["vector"].
  observations: OBSERVATIONS,
  // Bonus drops are the engine's own default, but they make an episode turn on
  // where a medkit happened to fall. Off is the cleaner thing to learn in.
  rules: { bonusDrops: 0 },
  weights: DEFAULT_WEIGHTS,
  progress: {},
  // A place each worm is paid to reach, as (env, agent) => ({x, y}) or null.
  // Off for a fight; on for the walking task that checks the pipeline learns.
  goals: null,
};

const NO_ACTION = { keys: 0, rope: 0, weapon: 0, fresh: false };

const range = (setting) => (Array.isArray(setting) ? setting : [setting, setting]);

export class WormEnv {
  constructor(engine, options = {}) {
    const settings = { ...DEFAULTS, ...options };
    if (!Number.isInteger(settings.agents) || settings.agents < 1) {
      throw new Error(`agents must be a whole number of at least 1, got ${settings.agents}`);
    }
    this.engine = engine;
    this.agents = settings.agents;
    this.spec = observationSpec({
      foeSlots: settings.observationFoes ?? this.agents - 1,
    });
    this.frameskip = settings.frameskip;
    this.episodeTicks = settings.episodeTicks;
    this.respawn = settings.respawn;
    this.terminateOnKill = settings.terminateOnKill;
    this.inputLatencyTicks = range(settings.inputLatencyTicks);
    this.loadout = settings.loadouts ?? settings.loadout;
    this.weights = settings.weights;
    this.observationKinds = settings.observations;
    this.makeGoal = settings.goals;
    this.reward = settings.reward ?? combatReward;
    // A fresh level per episode by default: one map teaches one map.
    this.makeLevel =
      typeof settings.level === "function"
        ? settings.level
        : settings.level
          ? () => settings.level
          : (engineIn, seed) => engineIn.randomLevel(seed, settings.levelOptions);
    this.world = engine.createWorld({ rules: settings.rules });
    // The engine knows who hit whom; this is where it says so.
    this.watch = watchDamage(this.world);
    this.seed = settings.seed ?? 1;
    this.worms = [];
    this.loadouts = [];
    this.views = [];
    this.observations = [];
    this.totals = [];
    this.events = emptyEvents(this.agents);
    this.progress = Array.from(
      { length: this.agents },
      () => new Progress(settings.progress),
    );
    this.alive = new Array(this.agents).fill(false);
    this.queues = [];
    this.latency = [];
    this.episode = 0;
    this.episodeSeed = null;
    this.episodeStartTick = 0;
    this.done = true;
  }

  /**
   * Start a match. Without a seed it takes the next one from the environment's
   * own generator, so a worker can loop `reset()` and still replay any episode
   * from the seed the returned info reports.
   */
  reset({ seed } = {}) {
    const episodeSeed = (seed ?? this.seed) >>> 0;
    this.seed = (episodeSeed + 0x9e3779b9) >>> 0;
    this.rng = makeRng(episodeSeed);
    this.episode++;
    this.episodeSeed = episodeSeed;

    // The world is emptied rather than rebuilt: its entity pools are thousands
    // of objects and an episode is over in milliseconds.
    this.world.reset(episodeSeed);
    this.world.level.Of(this.makeLevel(this.engine, episodeSeed));
    this.watch.clear();

    this.loadouts = Array.from({ length: this.agents }, (_, agent) =>
      this.loadout === "random"
        ? this.engine.randomLoadout(this.rng)
        : Array.isArray(this.loadout[0])
          ? this.loadout[agent]
          : this.loadout,
    );
    this.worms = this.loadouts.map((loadout, agent) =>
      this.engine.spawnWorm(this.world, {
        // The mod ships a fixed set of worm colours; past the end they repeat.
        color: agent % this.engine.wormColours,
        playerId: agent,
        loadout,
      }),
    );
    const [low, high] = this.inputLatencyTicks;
    // One delay per worm per episode: a connection does not change its mind
    // mid-match either.
    this.latency = this.worms.map(
      () => low + Math.floor(this.rng() * (high - low + 1)),
    );
    this.queues = this.latency.map((ticks) =>
      Array.from({ length: ticks }, () => NO_ACTION),
    );
    this.progress.forEach((progress, agent) =>
      progress.reset({ goal: this.makeGoal?.(this, agent) ?? null }),
    );
    this.alive = this.worms.map((worm) => Boolean(worm.u));
    this.totals = this.worms.map(() => ({}));
    this.episodeStartTick = this.world.qb;
    this.done = false;
    this.refreshViews();
    this.encodeObservations();
    return { observations: this.observations, info: this.info() };
  }

  /**
   * One decision per agent: a key bitmask, or the full `{ keys, rope, weapon }`
   * an action space with the rope in it needs.
   */
  step(actions) {
    if (this.done) throw new Error("the episode is over: call reset() first");
    if (actions.length !== this.agents) {
      throw new Error(`expected ${this.agents} actions, got ${actions.length}`);
    }
    for (const [agent, action] of actions.entries()) {
      const normalized = normalizeAction(action);
      const queue = this.queues[agent];
      // Held keys last the whole decision; the rope and weapon messages are
      // sent once, so only the first tick of the decision carries them.
      queue.push({ ...normalized, fresh: true });
      for (let tick = 1; tick < this.frameskip; tick++) queue.push(normalized);
    }

    this.watch.clear();
    for (let tick = 0; tick < this.frameskip; tick++) {
      for (let agent = 0; agent < this.agents; agent++) {
        const worm = this.worms[agent];
        const due = this.queues[agent].shift() ?? NO_ACTION;
        if (!worm.u) continue;
        applyNormalizedAction(this.world, worm, {
          keys: due.keys,
          rope: due.fresh ? due.rope : 0,
          weapon: due.fresh ? due.weapon : 0,
        });
      }
      this.world.update();
    }

    // Everything is read before anyone respawns, or a death would look like a
    // worm that healed back to full and moved across the map.
    this.refreshViews();
    tallyDamage(this.watch, this.agents, this.events);
    const rewards = [];
    const parts = [];
    for (let agent = 0; agent < this.agents; agent++) {
      const worm = this.worms[agent];
      const events = this.events[agent];
      events.died = this.alive[agent] && !worm.u ? 1 : 0;
      this.alive[agent] = Boolean(worm.u);
      const moved = this.progress[agent].update(
        worm.u ? worm : this.views[agent].self.position ?? worm,
        Boolean(worm.u),
      );
      const outcome = this.reward(events, moved, this.weights);
      rewards.push(outcome.reward);
      parts.push(outcome.parts);
      addEvents(this.totals[agent], events);
      addEvents(this.totals[agent], outcome.parts);
      this.totals[agent].reward = (this.totals[agent].reward ?? 0) + outcome.reward;
      this.totals[agent].stuckSteps =
        (this.totals[agent].stuckSteps ?? 0) + (moved.stuck ? 1 : 0);
      this.totals[agent].cellsVisited = moved.cellsVisited;
    }
    const killed = this.events.some((events) => events.killed > 0);

    let respawned = false;
    if (this.respawn) {
      for (const [agent, worm] of this.worms.entries()) {
        if (worm.u) continue;
        respawnWorm(this.world, worm, this.loadouts[agent]);
        // It is somewhere else entirely now; nothing about where it was holds.
        this.progress[agent].restart();
        this.alive[agent] = true;
        respawned = true;
      }
      if (respawned) this.refreshViews();
    }
    this.encodeObservations();
    this.done =
      this.world.qb - this.episodeStartTick >= this.episodeTicks ||
      (this.terminateOnKill && killed);
    return {
      observations: this.observations,
      rewards,
      done: this.done,
      info: { ...this.info(), events: this.events, parts },
    };
  }

  /** Every agent's view of the world as it stands. Cheap: no terrain is copied. */
  refreshViews() {
    this.views = this.worms.map((worm, agent) =>
      viewFromWorld(
        this.world,
        worm,
        this.worms.filter((_, other) => other !== agent),
      ),
    );
    return this.views;
  }

  /** The expensive half, so it runs once per step and not once per view. */
  encodeObservations() {
    this.observations = this.views.map((view, agent) =>
      observe(view, this.observations[agent] ?? {}, this.observationKinds, this.spec),
    );
    return this.observations;
  }

  info() {
    return {
      episode: this.episode,
      seed: this.episodeSeed,
      tick: this.world.qb,
      elapsedTicks: this.world.qb - this.episodeStartTick,
      map: this.world.level.name,
      agents: this.agents,
      observations: this.observationKinds,
      vectorSize: this.spec.vectorSize,
      inputLatencyTicks: this.latency,
      totals: this.totals,
    };
  }
}
