// The environment: worlds in, actions in, observations and rewards out.
//
// One instance is one match at a time — by default three worms in a free-for-all,
// which is a different game from a duel: the worm shooting at you is often not
// the one you are shooting at, and the third one is deciding which of you is
// worth interrupting. Everything that could make two runs of one seed differ
// goes through a single seeded generator, so a rollout is reproducible and a
// policy's bad episode can be replayed exactly.
import { KEYS, applyNormalizedAction, normalizeAction } from "./actions.js";
import { makeRng, respawnWorm, roomWeapons, watchDamage, weaponList } from "./engine.js";
import {
  AIM_RANGE_PX,
  MAP_CELLS,
  OBSERVATIONS,
  encodeMapTerrain,
  nearestFoe,
  observationSpec,
  observe,
  shotSolution,
} from "./observation.js";
import { Progress } from "./progress.js";
import {
  addEvents,
  combatReward,
  DEFAULT_WEIGHTS,
  emptyEvents,
  tallyDamage,
} from "./reward.js";
import { viewFromWorld } from "./view.js";

/** How wide the cone a worm is paid for aiming inside is. */
const AIM_CONE = Math.PI / 8;
/** A step bigger than this was a respawn, not a walk. */
const APPROACH_LIMIT_PX = 60;

// Stock Liero weapons that give a worm something to do at every range: shotgun,
// rifle, bazooka, mine, crackler. Pass "random" instead to draw five per worm
// per episode, which is what teaches the game rather than the shotgun.
export const DEFAULT_LOADOUT = [0, 2, 3, 5, 10];

/**
 * Weapons that do their damage where they hit, rather than by exploding.
 *
 * Half of a mod's weapons are explosives, and a policy that has not yet learned
 * to aim fires them at its own feet: at the start of a run, self-inflicted
 * damage outweighs damage dealt fifteen to one. What it learns from that is not
 * "aim better", it is "never fire" — measured, twice. Guns first, and widen the
 * pool once it can hit something.
 *
 * By name, not by id. An id means a different weapon in every mod — id 0 is the
 * shotgun in Liero 1.33 and the auto shotgun in Promode ReRevisited — so a list
 * of numbers is a list about one mod, and using it under another either picks
 * the wrong guns or runs off the end of a shorter list. Each entry is the
 * names one weapon goes by; the first that the loaded mod has is the one used.
 */
export const DIRECT_FIRE_NAMES = [
  ["SHOTGUN", "AUTO SHOTGUN"],
  ["CHAINGUN"],
  ["RIFLE"],
  ["WINCHESTER"],
  ["FLAMER", "FLAMETHROWER"],
  ["MINIGUN"],
  ["SUPER SHOTGUN"],
  ["HANDGUN"],
  ["ZIMM"],
  ["LASER"],
  ["UZI"],
  ["MINI ROCKETS"],
  ["DART", "DARTGUN"],
];

/** The ids those names have in whichever mod is loaded. */
export function directFire(engine) {
  const at = new Map(
    engine.settings.O.map((weapon, id) => [String(weapon.name).trim().toUpperCase(), id]),
  );
  const ids = [];
  for (const names of DIRECT_FIRE_NAMES) {
    for (const name of names) {
      if (at.has(name)) {
        ids.push(at.get(name));
        break;
      }
    }
  }
  return ids;
}

/**
 * Named pools a trainer can ask for.
 *
 * "room" is the weapons the watched room lets a worm spawn holding, read off
 * its own weapon screen; the rest of the mod still turns up in crates, which is
 * what the room's "Banned" actually means. "all" is every weapon there is.
 */
export const WEAPON_POOL_NAMES = ["starter", "room", "direct", "all"];

export const DEFAULTS = {
  // Three is a free-for-all, which is the interesting case; two is a duel and
  // more is a brawl. Nothing here is written for three in particular — the
  // observation, the attribution and the rewards are all built from this number.
  agents: 3,
  // How many other worms the vector describes. By default every one of them.
  // Pin it higher than `agents` and the same policy can play a duel and a
  // five-way without being retrained: the empty slots read as zeros.
  observationFoes: null,
  // Pixels per patch cell. The patch shows the same ground at any scale; a
  // larger number is fewer cells for the network to look at. See
  // `patchGeometry` for what each value comes to.
  patchScale: 2,
  // A second patch scale to encode as well, or null. Only an evaluation
  // seating two policies trained at different scales wants it.
  patchScale2: null,
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
  // Which weapons "random" draws from. Null is all of them.
  weaponPool: "direct",
  // Weapons nobody spawns holding, by name. They still appear in crates.
  banStart: [],
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
  // Decisions between re-reading the whole level for the map observation.
  // Reading it costs every pixel, so it is amortised: the terrain only changes
  // where somebody is digging, and four seconds of staleness at this scale is a
  // fraction of one cell.
  mapEvery: 60,
  // Decisions, per worm, over which the ladder terms fade to `shapingFloor`.
  // Zero leaves them at full strength for the whole run, which is what every
  // run has done so far. The trainer works this out from its step budget: it
  // knows how many worms are playing and this environment does not.
  shapingFullAt: 0,
  shapingFloor: 0,
  // Decisions this world is taken to have made before it started: a run
  // carrying on from a checkpoint hands over how far the ladder had already
  // faded, so the fade does not start again from the top.
  decisionsDone: 0,
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
      weaponFeatures: engine.weaponFeatures,
      ballistics: engine.ballistics,
      patchScale: settings.patchScale,
      patchScale2: settings.patchScale2 ?? null,
    });
    this.mapEvery = settings.mapEvery;
    this.shapingFullAt = Math.max(0, settings.shapingFullAt ?? 0);
    this.shapingFloor = settings.shapingFloor ?? 0;
    // How much of the ladder is still being paid. Counted in decisions taken
    // rather than episodes, because episodes are a clock and this is progress.
    this.decisions = Math.max(0, Math.trunc(settings.decisionsDone ?? 0));
    this.shaping = 1;
    if (this.shapingFullAt > 0 && this.decisions > 0) {
      const gone = Math.min(1, this.decisions / this.shapingFullAt);
      this.shaping = 1 - (1 - this.shapingFloor) * gone;
    }
    // Shared by every worm in this world: the terrain is the same for all of
    // them, and only where they are differs.
    this.mapTerrain = new Uint8Array(3 * MAP_CELLS);
    this.mapAt = -1;
    this.frameskip = settings.frameskip;
    this.episodeTicks = settings.episodeTicks;
    this.respawn = settings.respawn;
    this.terminateOnKill = settings.terminateOnKill;
    this.inputLatencyTicks = range(settings.inputLatencyTicks);
    this.loadout = settings.loadouts ?? settings.loadout;
    if (typeof settings.weaponPool === "string" && !WEAPON_POOL_NAMES.includes(settings.weaponPool)) {
      throw new Error(
        `unknown weapon pool ${settings.weaponPool}: expected ${WEAPON_POOL_NAMES.join(", ")}`,
      );
    }
    // "all" is null, which is every weapon — so the name has to be looked up
    // rather than defaulted through, or asking for all of them reads as a typo.
    const chosen = this.#poolNamed(settings.weaponPool, settings.slots ?? 5);
    const barred = this.#bannedAtStart(settings.banStart);
    // Null means every weapon, so the ban has to be spelled out as a list.
    this.weaponPool = !barred.size
      ? chosen
      : (chosen ?? this.engine.settings.O.map((_, id) => id)).filter((id) => !barred.has(id));
    // A partial set of weights overrides the defaults rather than replacing
    // them, so a trainer can turn one knob without restating the rest.
    this.weights = { ...DEFAULT_WEIGHTS, ...(settings.weights ?? {}) };
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
    // Each worm's previous decision, for the observation. Null until it has
    // made one this match.
    this.lastActions = [];
    this.episode = 0;
    this.episodeSeed = null;
    this.episodeStartTick = 0;
    this.done = true;
    // True on the one step whose observation is an episode's last.
    this.ending = false;
  }

  /**
   * Start a match. Without a seed it takes the next one from the environment's
   * own generator, so a worker can loop `reset()` and still replay any episode
   * from the seed the returned info reports.
   */
  /** The ids a named weapon pool stands for, or null for every weapon. */
  #poolNamed(pool, slots) {
    if (typeof pool !== "string") return pool;
    const { settings, mod } = { settings: this.engine.settings, mod: this.engine.mod };
    if (pool === "starter") {
      const starter = weaponList(settings, mod, "starter-weapons.txt");
      if (starter) return starter;
      // The same silent fall-through as the room list: a curriculum that was
      // asked for and quietly not given is a run whose log lies about itself.
      console.warn(
        `no starter-weapons.txt beside ${mod}: "starter" was asked for but ` +
          `every one of ${settings.O.length} weapons can be spawned with. ` +
          "Write the list, or pass --weapons room.",
      );
      return null;
    }
    if (pool === "room") {
      const room = roomWeapons(settings, mod);
      if (room) return room.enabled;
      // Falling through here silently is how a run ends up spawning worms with
      // the weapons a room only ever puts in a crate, while its own log still
      // says the room's list was used.
      console.warn(
        `no room-weapons.txt beside ${mod}: every one of ${settings.O.length} ` +
          "weapons can be spawned with, crate-only ones included. " +
          "Run `npm run mods` to lay the room's list down.",
      );
      return null;
    }
    if (pool === "direct") return this.#directOrEverything(slots);
    return null; // "all"
  }

  /**
   * The ids of weapons named in `banStart`, for dropping from a loadout.
   *
   * Only from the loadout. Crates are filled by the engine from the mod's whole
   * list, so a weapon barred here still turns up as a bonus now and then —
   * which is how the rooms run it: the awkward ones are something you find,
   * not something you spawn holding.
   */
  #bannedAtStart(names) {
    if (!names?.length) return new Set();
    const wanted = new Set(names.map((name) => String(name).trim().toUpperCase()));
    const ids = new Set();
    this.engine.settings.O.forEach((weapon, id) => {
      if (wanted.has(String(weapon.name).trim().toUpperCase())) ids.add(id);
    });
    const missed = [...wanted].filter(
      (name) => !this.engine.settings.O.some((w) => String(w.name).trim().toUpperCase() === name),
    );
    if (missed.length) {
      console.warn(
        `no weapon called ${missed.join(", ")} in ${this.engine.settings.name}; ` +
          "check the spelling against the mod's own names",
      );
    }
    return ids;
  }

  /**
   * The guns, when this mod's guns can be told apart by name; otherwise all of
   * them, loudly.
   *
   * "Direct fire" is a list of weapons this project knows to be safe for a
   * policy that cannot aim yet. Under a community mod it is a list of names
   * nothing matches — CS Rewormed's hundred and twenty-nine weapons share not
   * one name with Liero's forty. Falling back silently would train on
   * explosives while the run log still said "direct", so it says so instead.
   */
  #directOrEverything(slots) {
    const ids = directFire(this.engine);
    if (ids.length >= slots) return ids;
    console.warn(
      `the direct-fire weapons cannot be named in ${this.engine.settings.name}: ` +
        `${ids.length} of them matched, and a loadout needs ${slots}. ` +
        "Training on every weapon instead.",
    );
    return null;
  }

  /**
   * Whether this worm is pointing at somebody it could actually hit.
   *
   * Aiming pays nothing on its own in this game: the reward for it arrives
   * later, as damage, if the shot lands at all — and a policy that cannot aim
   * never fires well enough to find that out. So the alignment itself is paid
   * for, but only when it is real: a living foe, within reach, with nothing
   * solid in between. Pointing at a wall earns nothing, and neither does
   * pointing at somebody across the map through a hill.
   */
  #aimAt(agent) {
    const view = this.views[agent];
    const self = view?.self;
    const was = this.foeRange[agent];
    this.foeRange[agent] = null;
    const nothing = { onTarget: 0, aimedShot: 0, approach: 0 };
    if (!self?.alive) return nothing;
    const foe = nearestFoe(view);
    if (!foe) return nothing;
    const dx = foe.position.x - self.position.x;
    const dy = foe.position.y - self.position.y;
    const range = Math.hypot(dx, dy);
    this.foeRange[agent] = { id: foe.id, range };
    // Ground closed on the nearest foe since the last decision. Signed, so
    // backing away costs exactly what closing in pays and a worm cannot farm it
    // by pacing in and out. Only against the same foe, and only over a step
    // small enough to have been walked: a respawn moves a worm across the map.
    const approach =
      was && was.id === foe.id && Math.abs(was.range - range) < APPROACH_LIMIT_PX
        ? was.range - range
        : 0;
    if (range < 1 || range > AIM_RANGE_PX) return { ...nothing, approach };

    // The shot this weapon would actually have to make — its arc, not the
    // straight line, which is the wrong aim for thirty-seven of the forty-five
    // weapons a worm starts with. The same solution goes into the foe's slot
    // of the observation, so what is paid for is what the policy is shown.
    const solution = shotSolution(view, foe, this.engine.ballistics);
    if (!solution || !solution.clear || Math.abs(solution.off) > AIM_CONE) {
      return { ...nothing, approach };
    }
    // Closer to the middle of the cone is worth more, so there is a gradient to
    // climb rather than a cliff to find.
    const onTarget = 1 - Math.abs(solution.off) / AIM_CONE;
    return { onTarget, aimedShot: this.firing[agent] ? onTarget : 0, approach };
  }

  /**
   * Five of one weapon, a different one each episode.
   *
   * A worm handed five weapons it has never used, in a fight, learns nothing
   * about any of them: whatever it was holding when something good happened
   * gets the credit. One weapon for a whole episode is long enough to find out
   * what that weapon does — how far it carries, how much it drops, what it does
   * to whoever fired it — before any of that has to be chosen between.
   */
  #drillLoadout() {
    // The same weapon for everybody, so the whole episode is about that one
    // weapon: both sides learn what it does and what it does back.
    return [0, 0, 0, 0, 0].map(() => this.drillWeapon);
  }

  /** Pick the episode's weapon, once, before anybody is given a loadout. */
  #chooseDrillWeapon() {
    const pool = this.weaponPool ?? this.engine.settings.O.map((_, id) => id);
    this.drillWeapon = pool[Math.floor(this.rng() * pool.length)];
  }

  /** The five weapons one worm starts an episode holding. */
  #loadoutFor(agent) {
    if (this.loadout === "drill") return this.#drillLoadout();
    if (this.loadout === "random") {
      return this.engine.randomLoadout(this.rng, { pool: this.weaponPool });
    }
    return Array.isArray(this.loadout[0]) ? this.loadout[agent] : this.loadout;
  }

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
    // A new level is a new map; nothing cached about the last one holds.
    this.mapAt = -1;

    if (this.loadout === "drill") this.#chooseDrillWeapon();
    this.loadouts = Array.from({ length: this.agents }, (_, agent) =>
      this.#loadoutFor(agent),
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
    this.lastActions = this.worms.map(() => null);
    this.progress.forEach((progress, agent) => {
      // A new level every episode, and covering it is paid as a share of it.
      progress.sized(this.world.level);
      progress.reset({ goal: this.makeGoal?.(this, agent) ?? null });
    });
    this.alive = this.worms.map((worm) => Boolean(worm.u));
    this.firing = this.worms.map(() => false);
    this.foeRange = this.worms.map(() => null);
    this.totals = this.worms.map(() => ({}));
    this.episodeStartTick = this.world.qb;
    this.done = false;
    this.ending = false;
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
    this.ending = false;
    if (actions.length !== this.agents) {
      throw new Error(`expected ${this.agents} actions, got ${actions.length}`);
    }
    for (const [agent, action] of actions.entries()) {
      const normalized = normalizeAction(action);
      // Whether it pulled the trigger this decision AND had something to fire,
      // for the aim reward. Firing while lined up is the thing worth paying
      // for, and the view does not say. The gun has to be loaded: the term was
      // paid for holding the key down, so a worm with an empty slot could stand
      // lined up on somebody with the trigger held and collect the whole term
      // for the rest of the episode without a shot leaving the barrel. Read
      // before the step, which is when the decision was made — with input
      // latency the shot itself lands some ticks later, so this is "it could
      // have fired", not "it did".
      const holding = this.views[agent]?.self;
      const loaded = holding?.weapons?.[holding.selectedWeapon];
      this.firing[agent] =
        (normalized.keys & KEYS.fire) !== 0 &&
        Boolean(loaded) &&
        loaded.ammo > 0 &&
        loaded.cooldownTicksRemaining <= 0;
      this.lastActions[agent] = normalized;
      const queue = this.queues[agent];
      // Held keys last the whole decision; the rope and weapon messages are
      // sent once, so only the first tick of the decision carries them.
      queue.push({ ...normalized, fresh: true });
      for (let tick = 1; tick < this.frameskip; tick++) queue.push(normalized);
    }

    this.decisions++;
    if (this.shapingFullAt > 0) {
      const gone = Math.min(1, this.decisions / this.shapingFullAt);
      this.shaping = 1 - (1 - this.shapingFloor) * gone;
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
      const outcome = this.reward(
        events,
        { ...moved, ...this.#aimAt(agent) },
        this.weights,
        this.shaping,
      );
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

    // Which worms came back this step. Reported alongside the observation,
    // because whoever is learning from a memory of where it was has to be told
    // that where it was no longer holds: it is somewhere else, with full health
    // and a fresh loadout, and the last few seconds were somebody else's life.
    const respawned = new Array(this.agents).fill(false);
    if (this.respawn) {
      for (const [agent, worm] of this.worms.entries()) {
        if (worm.u) continue;
        respawnWorm(this.world, worm, this.loadouts[agent]);
        // It is somewhere else entirely now; nothing about where it was holds.
        this.progress[agent].restart();
        this.alive[agent] = true;
        respawned[agent] = true;
      }
      if (respawned.some(Boolean)) this.refreshViews();
    }
    this.encodeObservations();
    // Nothing in this environment ever really ends. Worms respawn, the world
    // keeps running, and what stops an episode is a clock this project set for
    // its own convenience — so every ending here is a truncation, and the
    // observation being returned is the last one of the episode rather than
    // the first one of the next. Whoever is learning from this has to value
    // that state rather than treat it as worth nothing, which is why the reset
    // waits for the next call instead of happening here.
    this.done =
      this.world.qb - this.episodeStartTick >= this.episodeTicks ||
      (this.terminateOnKill && killed);
    this.ending = this.done;
    return {
      observations: this.observations,
      rewards,
      done: this.done,
      // The episode is over and this is its final state; `reset()` has not run.
      truncated: this.done,
      respawned,
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
        this.latency[agent] ?? 0,
        this.lastActions[agent] ?? null,
      ),
    );
    return this.views;
  }

  /** The expensive half, so it runs once per step and not once per view. */
  encodeObservations() {
    if (this.observationKinds.includes("map")) {
      const now = this.world.qb;
      if (this.mapAt < 0 || now - this.mapAt >= this.mapEvery * this.frameskip) {
        encodeMapTerrain(this.views[0].terrain, this.mapTerrain);
        this.mapAt = now;
      }
    }
    this.observations = this.views.map((view, agent) =>
      observe(view, this.observations[agent] ?? {}, this.observationKinds, this.spec, this.mapTerrain),
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
      shaping: this.shaping,
      totals: this.totals,
    };
  }
}
