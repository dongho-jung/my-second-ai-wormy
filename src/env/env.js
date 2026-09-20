// The environment: worlds in, actions in, observations and rewards out.
//
// One instance is one match at a time. `reset` starts a new one, `step` hands
// every agent its action and advances the world `frameskip` ticks. Everything
// that could make two runs of one seed differ goes through a single seeded
// generator, so a rollout is reproducible and a policy's bad episode can be
// replayed exactly.
import { applyNormalizedAction, normalizeAction } from "./actions.js";
import { makeRng, respawnWorm } from "./engine.js";
import { OBSERVATIONS, observe } from "./observation.js";
import { addEvents, combatReward, DEFAULT_WEIGHTS, scoreOf } from "./reward.js";
import { viewFromWorld } from "./view.js";

// Stock Liero weapons that give a worm something to do at every range: shotgun,
// rifle, bazooka, mine, crackler.
export const DEFAULT_LOADOUT = [0, 2, 3, 5, 10];

export const DEFAULTS = {
  agents: 2,
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
  loadout: DEFAULT_LOADOUT,
  // Both pictures by default. The terrain patch is about ten times the cost of
  // the vector, so a task that does not need it — walking somewhere, a first
  // check that the pipeline learns anything at all — should pass ["vector"].
  observations: OBSERVATIONS,
  // Bonus drops are the engine's own default, but they make an episode turn on
  // where a medkit happened to fall. Off is the cleaner thing to learn in.
  rules: { bonusDrops: 0 },
  weights: DEFAULT_WEIGHTS,
};

const NO_ACTION = { keys: 0, rope: 0, weapon: 0, fresh: false };

const range = (setting) => (Array.isArray(setting) ? setting : [setting, setting]);

export class WormEnv {
  constructor(engine, options = {}) {
    const settings = { ...DEFAULTS, ...options };
    this.engine = engine;
    this.agents = settings.agents;
    this.frameskip = settings.frameskip;
    this.episodeTicks = settings.episodeTicks;
    this.respawn = settings.respawn;
    this.terminateOnKill = settings.terminateOnKill;
    this.inputLatencyTicks = range(settings.inputLatencyTicks);
    this.loadouts =
      settings.loadouts ??
      Array.from({ length: this.agents }, () => settings.loadout);
    this.weights = settings.weights;
    this.observationKinds = settings.observations;
    this.reward = settings.reward ?? combatReward;
    // A fresh level per episode by default: one map teaches one map.
    this.makeLevel =
      typeof settings.level === "function"
        ? settings.level
        : settings.level
          ? () => settings.level
          : (engineIn, seed) => engineIn.randomLevel(seed, settings.levelOptions);
    this.world = engine.createWorld({ rules: settings.rules });
    this.seed = settings.seed ?? 1;
    this.worms = [];
    this.views = [];
    this.observations = [];
    this.scores = [];
    this.totals = [];
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

    this.worms = Array.from({ length: this.agents }, (_, agent) =>
      this.engine.spawnWorm(this.world, {
        color: agent,
        playerId: agent,
        loadout: this.loadouts[agent],
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
    this.totals = this.worms.map(() => ({}));
    this.episodeStartTick = this.world.qb;
    this.done = false;
    this.refreshViews();
    this.scores = this.views.map(scoreOf);
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

    // Read the reward before anyone respawns, or a death looks like a worm that
    // healed back to full.
    this.refreshViews();
    const outcomes = this.views.map((view, agent) =>
      this.reward(this.scores[agent], scoreOf(view), this.weights),
    );
    for (const [agent, outcome] of outcomes.entries()) {
      addEvents(this.totals[agent], outcome.events);
    }
    const killed = outcomes.some((outcome) => outcome.events.killed > 0);

    let respawned = false;
    if (this.respawn) {
      for (const [agent, worm] of this.worms.entries()) {
        if (worm.u) continue;
        respawnWorm(this.world, worm, this.loadouts[agent]);
        respawned = true;
      }
      if (respawned) this.refreshViews();
    }
    this.scores = this.views.map(scoreOf);
    this.encodeObservations();
    this.done =
      this.world.qb - this.episodeStartTick >= this.episodeTicks ||
      (this.terminateOnKill && killed);
    return {
      observations: this.observations,
      rewards: outcomes.map((outcome) => outcome.reward),
      done: this.done,
      info: {
        ...this.info(),
        events: outcomes.map((outcome) => outcome.events),
      },
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
      observe(view, this.observations[agent] ?? {}, this.observationKinds),
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
      observations: this.observationKinds,
      inputLatencyTicks: this.latency,
      totals: this.totals,
    };
  }
}
