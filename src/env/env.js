// The environment: worlds in, actions in, observations and rewards out.
//
// One instance is one match at a time — by default three worms in a free-for-all,
// which is a different game from a duel: the worm shooting at you is often not
// the one you are shooting at, and the third one is deciding which of you is
// worth interrupting. Everything that could make two runs of one seed differ
// goes through a single seeded generator, so a rollout is reproducible and a
// policy's bad episode can be replayed exactly.
import { KEYS, ROPE, RopeHold, applyNormalizedAction, normalizeAction } from "./actions.js";
import { makeRng, respawnWorm, roomWeapons, watchDamage, weaponList } from "./engine.js";
import {
  AIM_RANGE_PX,
  MAP_TERRAIN_SIZE,
  OBSERVATIONS,
  encodeMapTerrain,
  nearestFoe,
  observationSpec,
  observe,
  shotSolution,
} from "./observation.js";
import { GoalCurriculum, GoalDeadlineCurriculum } from "./curriculum.js";
import { Progress } from "./progress.js";
import {
  addEvents,
  combatReward,
  DEFAULT_WEIGHTS,
  emptyEvents,
  MOVEMENT_WEIGHTS,
  tallyDamage,
} from "./reward.js";
import { viewFromWorld } from "./view.js";
import { BACKGROUND, DIGGABLE, SHOT_STOPS, solidAt } from "./terrain.js";

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
  // How far from the worm "random" draws its goal, in pixels: a number, or
  // [from, to] that grows from the first to the second over `goalRadiusFullAt`
  // decisions. Null draws from the whole map, which is what the first movement
  // runs did — and on these maps that is a destination 450-550 px away on
  // average, fourteen seconds of walking on open ground and a rope everywhere
  // else, handed to a policy that cannot walk yet. A run that starts within a
  // hundred pixels learns to walk and jump first and meets the rope when the
  // goals move out of reach.
  goalRadiusPx: null,
  goalRadiusFullAt: 0,
  // What moves the radius from the first number to the second. "steps" grows
  // it with the decisions taken, over `goalRadiusFullAt` of them. "success"
  // grows it when the worms have been reaching their destinations and brings
  // it back when they have not — see curriculum.js, which also says why the
  // clock was the wrong thing to move it with.
  goalRadiusMode: "steps",
  // Where a "success" curriculum starts, for a run carrying on from a
  // checkpoint that had got somewhere. Null starts at the first number.
  goalRadiusStart: null,
  // Its window and thresholds, laid over CURRICULUM_DEFAULTS.
  goalCurriculum: null,
  // Decisions a goal is kept before it is given up on and another handed out.
  // 0 keeps it for the whole episode. A worm sent somewhere it cannot get to
  // otherwise spends the rest of its minute learning nothing.
  goalPatience: 0,
  // How many destinations one worm receives in an episode. Zero is unlimited.
  // A fixed benchmark uses one: every policy then attempts the same task once
  // instead of a lucky sequence of short goals paying over and over.
  goalsPerEpisode: 0,
  // End once every worm has resolved that quota. Training can immediately
  // draw a fresh task instead of filling the rest of the clock with no goal.
  // Validation leaves this off so every policy gets the same fixed horizon.
  endOnGoals: false,
  // When set below goalPatience, success tightens the deadline toward this
  // value and failure loosens it again. Off by default.
  goalPatienceMin: null,
  goalPatienceStart: null,
  goalDeadlineCurriculum: null,
  // Weights laid over `weights` after it is resolved, so a run can turn one
  // knob (`--rope-throw-cost`, `--suicide-cost`) without restating a table.
  weightOverrides: null,
  // Ignore the fire key and weapon switching. For stages where the lesson is
  // getting somewhere: see where this is read in `step`.
  lockWeapons: false,
  // Decisions between one rope message and the next being listened to. 0 lets
  // every decision carry one, which at maximum entropy means throwing five
  // times a second and letting go five times a second.
  //
  // What that costs is not what it looks like. Measured with a random policy
  // over three six-worm episodes, per worm:
  //
  //   cooldown   throws   decisions held   share of the episode held
  //          0    301.9            448.1                       49.8%
  //          5     82.6            456.0                       50.7%
  //         10     42.6            444.6                       49.4%
  //         30     15.1            447.8                       49.8%
  //
  // A thrown rope attaches within one decision, and throwing and releasing at
  // the same rate leaves a worm attached about half the time whatever this is
  // set to. So the cost of thrashing is not time spent off the rope. What
  // changes is how long one throw lasts — 1.5 decisions at 0, 10.5 at 10 — and
  // therefore whether a rope is a state the policy can act from or one that
  // has already changed by the time the next decision lands.
  //
  // Whether that matters to learning is what the a/b is for. It is not settled
  // by the table above.
  ropeCooldown: 0,
  // Decisions a rope throw is committed to: another throw is ignored and the
  // jump key dropped, so the rope gets to pull. See RopeHold in actions.js.
  // 0 leaves every decision free, which is what every run before 2026-09-23
  // had, and none of them held a rope for more than two decisions.
  ropeHold: 0,
  // What share of destinations are drawn above the worm, and by how much at
  // least. Half the goals a jump cannot reach means the success curriculum
  // cannot move on until the rope is being used — walking and jumping reach
  // the other half and no more.
  goalAboveShare: 0,
  goalAbovePx: 48,
  // What share deliberately have solid terrain across the straight line from
  // start to destination. These are the tasks that teach going around a ledge
  // instead of blindly following the goal arrow into its underside.
  goalDetourShare: 0,
  // What share are buried: every open pixel within a few dozen of the goal is
  // turned to dirt when it is handed out, so the only way in is to dig. The
  // goals above are all somewhere a worm can already stand, and in the room's
  // maps nearly every one of them is reachable without touching the dig key,
  // so without these a policy is never asked to dig at all.
  goalDigShare: 0,
  // `signed` pays every pixel closer and charges every pixel farther. `best`
  // pays only when the worm beats its closest distance so far: a necessary
  // detour is neutral, while pacing over the same ground cannot farm reward.
  goalProgressMode: "signed",
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
  // Distance-to-goal is scaffolding too: useful while a policy is learning
  // which way to go, but indifferent to a slow or winding route once it gets
  // there. A speed fine-tune can fade just this term while leaving the arrival
  // and speed rewards intact. The schedule begins at goalProgressStartAt so a
  // resumed run can start a fresh fade instead of inheriting the old clock.
  goalProgressFullAt: 0,
  goalProgressFloor: 0,
  goalProgressStartAt: 0,
  // Decisions this world is taken to have made before it started: a run
  // carrying on from a checkpoint hands over how far the ladder had already
  // faded, so the fade does not start again from the top.
  decisionsDone: 0,
};

const NO_ACTION = { keys: 0, rope: 0, weapon: 0, fresh: false };

const range = (setting) => (Array.isArray(setting) ? setting : [setting, setting]);

// How far under a candidate point the ground may be. A goal floating in the
// middle of a cavern is reachable only by rope, and asking for that before the
// worm can walk is the wrong lesson first.
const GOAL_GROUND_PX = 48;

/**
 * A random spot a worm could stand: background, with something solid not far
 * below it. Returns null rather than looping forever on a map where that is
 * hard to find — a worm with no goal simply has no goal that episode.
 */
export function groundedGoal(terrain, rng, tries = 64) {
  if (!terrain) return null;
  for (let attempt = 0; attempt < tries; attempt++) {
    const x = Math.floor(rng() * terrain.width);
    const y = Math.floor(rng() * terrain.height);
    if (solidAt(terrain, x, y)) continue;
    for (let below = 1; below <= GOAL_GROUND_PX; below++) {
      if (solidAt(terrain, x, y + below)) return { x, y };
    }
  }
  return null;
}

// A buried goal's ball of dirt: at most this wide, at least this much wider
// than the arrival circle so digging is unavoidable, and never reaching within
// this far of where the worm starts.
const DIG_BALL_PX = 44;
const DIG_MIN_PX = 32;
const DIG_CLEARANCE_PX = 30;

/** Closer than this and the worm has arrived before it has been told to go. */
const GOAL_MIN_PX = 40;

/**
 * Whether the straight route crosses a real thickness of solid terrain.
 *
 * The first and last few pixels are ignored: the worm starts on ground and a
 * valid goal has ground below it, neither of which makes the route a detour.
 * A run of several solid pixels catches walls and ledges without classifying a
 * one-pixel fleck of dirt as a navigation problem.
 */
export function directRouteBlocked(terrain, from, to, { edgePx = 12, solidRunPx = 6 } = {}) {
  if (!terrain || !from || !to) return false;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= edgePx * 2 + solidRunPx) return false;
  let run = 0;
  for (let along = edgePx; along <= distance - edgePx; along++) {
    const share = along / distance;
    if (solidAt(terrain, from.x + dx * share, from.y + dy * share)) {
      run++;
      if (run >= solidRunPx) return true;
    } else {
      run = 0;
    }
  }
  return false;
}

/**
 * The same, within `radius` pixels of `from`. Drawn from the square around the
 * point and kept when inside the circle and at least a few strides away, so a
 * short radius is a short walk and not a spot underfoot. Null when nothing
 * standable is that close, which a caller falls back from.
 */
export function groundedGoalNear(
  terrain,
  rng,
  from,
  radius,
  { tries = 64, abovePx = 0, blocked = false } = {},
) {
  if (!terrain || !from || !(radius > GOAL_MIN_PX)) return null;
  const left = Math.max(0, Math.floor(from.x - radius));
  const right = Math.min(terrain.width - 1, Math.ceil(from.x + radius));
  const top = Math.max(0, Math.floor(from.y - radius));
  // Asked for a spot above the worm, only rows that far up are drawn from.
  const bottom = Math.min(terrain.height - 1, Math.ceil(from.y + radius), Math.floor(from.y - abovePx));
  if (bottom < top) return null;
  for (let attempt = 0; attempt < tries; attempt++) {
    const x = left + Math.floor(rng() * (right - left + 1));
    const y = top + Math.floor(rng() * (bottom - top + 1));
    const distance = Math.hypot(x - from.x, y - from.y);
    if (distance > radius || distance < GOAL_MIN_PX) continue;
    if (solidAt(terrain, x, y)) continue;
    for (let below = 1; below <= GOAL_GROUND_PX; below++) {
      if (solidAt(terrain, x, y + below)) {
        const goal = { x, y };
        if (!blocked || directRouteBlocked(terrain, from, goal)) return goal;
        break;
      }
    }
  }
  return null;
}

/**
 * What `weights` means. A name, because a worker is configured over JSON and a
 * table of numbers does not want to be typed out on a command line. A table
 * overrides the defaults rather than replacing them, so a trainer can turn one
 * knob without restating the rest.
 */
function weightsFor(weights) {
  if (weights === "movement") return MOVEMENT_WEIGHTS;
  if (weights === "fight" || weights == null) return DEFAULT_WEIGHTS;
  return { ...DEFAULT_WEIGHTS, ...weights };
}

/**
 * What `goals` means. A function is used as it is; "random" is the built-in
 * above, which is what a command line can ask for. Anything else is no goals,
 * which is every run that came before this setting existed.
 */
function goalMaker(goals) {
  if (typeof goals === "function") return goals;
  if (goals === "random") {
    return (env, agent) => {
      const terrain = env.views[agent]?.terrain;
      const radius = env.goalRadius();
      if (radius !== null) {
        const from = env.views[agent]?.self?.position;
        // Draw this choice once. Retrying candidates must not silently change
        // the requested class just because the first point was unsuitable.
        const wantsDetour = env.goalDetourShare > 0 && env.rng() < env.goalDetourShare;
        const wantsAbove = env.goalAboveShare > 0 && env.rng() < env.goalAboveShare;
        // Drawn only when asked for, so runs without buried goals keep the
        // exact sequence of destinations they always had.
        const wantsDig = env.goalDigShare > 0 && env.rng() < env.goalDigShare;
        const mark = (goal) => (goal && wantsDig ? { ...goal, dig: true } : goal);
        if (wantsDetour) {
          const around = groundedGoalNear(terrain, env.rng, from, radius, {
            tries: 192,
            abovePx: wantsAbove ? env.goalAbovePx : 0,
            blocked: true,
          });
          if (around) return mark({ ...around, detour: true });
        }
        // Some of the time, somewhere a jump does not reach; the rest of the
        // time anywhere within reach, which may still be up.
        if (wantsAbove) {
          const up = groundedGoalNear(terrain, env.rng, from, radius, { abovePx: env.goalAbovePx });
          if (up) return mark(up);
        }
        const near = groundedGoalNear(terrain, env.rng, from, radius);
        if (near) return mark(near);
      }
      return groundedGoal(terrain, env.rng);
    };
  }
  return null;
}

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
    this.goalProgressFullAt = Math.max(0, settings.goalProgressFullAt ?? 0);
    this.goalProgressFloor = settings.goalProgressFloor ?? 0;
    this.goalProgressStartAt = Math.max(0, Math.trunc(settings.goalProgressStartAt ?? 0));
    this.goalProgressScale = 1;
    this.goalProgressMode = settings.goalProgressMode ?? "signed";
    if (!["signed", "best"].includes(this.goalProgressMode)) {
      throw new Error(`goalProgressMode must be signed or best, got ${this.goalProgressMode}`);
    }
    if (this.goalProgressFullAt > 0) {
      const gone = Math.min(
        1,
        Math.max(0, this.decisions - this.goalProgressStartAt) / this.goalProgressFullAt,
      );
      this.goalProgressScale = 1 - (1 - this.goalProgressFloor) * gone;
    }
    // Shared by every worm in this world: the terrain is the same for all of
    // them, and only where they are differs.
    this.mapTerrain = new Uint8Array(MAP_TERRAIN_SIZE);
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
    this.weights = { ...weightsFor(settings.weights), ...(settings.weightOverrides ?? {}) };
    this.observationKinds = settings.observations;
    this.makeGoal = goalMaker(settings.goals);
    this.goalRadiusPx = settings.goalRadiusPx ?? null;
    this.goalRadiusFullAt = Math.max(0, settings.goalRadiusFullAt ?? 0);
    this.goalRadiusMode = settings.goalRadiusMode ?? "steps";
    if (!["steps", "success"].includes(this.goalRadiusMode)) {
      throw new Error(`goalRadiusMode must be steps or success, got ${this.goalRadiusMode}`);
    }
    // One per world, across its episodes: the destinations its worms reach
    // and give up on are what move it, and a new match does not forget them.
    this.curriculum =
      this.goalRadiusMode === "success" && Array.isArray(this.goalRadiusPx)
        ? new GoalCurriculum({
            from: this.goalRadiusPx[0],
            to: this.goalRadiusPx[1],
            start: settings.goalRadiusStart ?? null,
            ...(settings.goalCurriculum ?? {}),
          })
        : null;
    this.goalPatience = Math.max(0, Math.trunc(settings.goalPatience ?? 0));
    this.goalsPerEpisode = Math.max(0, Math.trunc(settings.goalsPerEpisode ?? 0));
    this.endOnGoals = Boolean(settings.endOnGoals);
    this.deadlineCurriculum =
      settings.goalPatienceMin !== null &&
      settings.goalPatienceMin !== undefined &&
      this.goalPatience > 0
        ? new GoalDeadlineCurriculum({
            from: this.goalPatience,
            to: settings.goalPatienceMin,
            start: settings.goalPatienceStart ?? null,
            ...(settings.goalDeadlineCurriculum ?? {}),
          })
        : null;
    this.lockWeapons = Boolean(settings.lockWeapons);
    this.ropeCooldown = Math.max(0, Math.trunc(settings.ropeCooldown ?? 0));
    this.ropeHold = Math.max(0, Math.trunc(settings.ropeHold ?? 0));
    this.goalAboveShare = Math.min(1, Math.max(0, settings.goalAboveShare ?? 0));
    this.goalAbovePx = Math.max(0, settings.goalAbovePx ?? 0);
    this.goalDetourShare = Math.min(1, Math.max(0, settings.goalDetourShare ?? 0));
    this.goalDigShare = Math.min(1, Math.max(0, settings.goalDigShare ?? 0));
    // A material that is dirt and nothing else: solid to a worm, holds a rope,
    // and goes away when dug. Buried goals are packed with it.
    this.dirtIndex = Array.from(engine.materialFlags).findIndex(
      (flags) => !(flags & BACKGROUND) && (flags & DIGGABLE) && !(flags & SHOT_STOPS),
    );
    this.makeStart = typeof settings.starts === "function" ? settings.starts : null;
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
    this.terminated = false;
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
    if (this.makeStart) {
      this.worms.forEach((worm, agent) => {
        const start = this.makeStart(this, agent);
        if (!start) return;
        worm.x = start.x;
        worm.y = start.y;
        worm.f = 0;
        worm.b = 0;
        worm.Wa = 0;
        worm.Fa.Sc = false;
        worm.Fa.jc = false;
      });
    }
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
    this.progress.forEach((progress) => {
      // A new level every episode, and covering it is paid as a share of it.
      progress.sized(this.world.level);
      progress.reset();
    });
    this.alive = this.worms.map((worm) => Boolean(worm.u));
    this.firing = this.worms.map(() => false);
    // Whether each worm threw the rope this decision, for the reward.
    this.ropeThrown = this.worms.map(() => 0);
    // The decision each worm's next rope message will be listened to on.
    this.ropeReadyAt = this.worms.map(() => 0);
    // Each worm's commitment to its last throw.
    this.ropeHolds = this.worms.map(() => new RopeHold(this.ropeHold));
    this.foeRange = this.worms.map(() => null);
    this.totals = this.worms.map(() => ({}));
    this.episodeStartTick = this.world.qb;
    this.done = false;
    this.terminated = false;
    this.ending = false;
    this.refreshViews();
    // Goals are drawn after the views exist, because picking a spot means
    // reading the terrain and the views are what carry it.
    this.assignGoals();
    this.encodeObservations();
    return { observations: this.observations, info: this.info() };
  }

  /**
   * How far a goal may be drawn from the worm right now, or null for anywhere.
   * A range grows with the decisions this world has made, the way the ladder
   * fades: a resumed run carries on from where the radius had got to.
   */
  goalRadius() {
    const setting = this.goalRadiusPx;
    if (setting === null || setting === undefined) return null;
    if (!Array.isArray(setting)) return setting;
    if (this.curriculum) return this.curriculum.radius;
    const [from, to] = setting;
    if (!(this.goalRadiusFullAt > 0)) return to;
    const gone = Math.min(1, this.decisions / this.goalRadiusFullAt);
    return from + (to - from) * gone;
  }

  /** Decisions the current destination gets before it is replaced. */
  goalDeadline() {
    return this.deadlineCurriculum?.patience ?? this.goalPatience;
  }

  /**
   * Pack every open pixel within `DIG_BALL_PX` of the goal with dirt, leaving
   * rock alone. The arrival circle is smaller than the ball, so the worm has
   * to dig its way in. Refused when the worm would be inside the ball itself
   * or the ball would not be larger than the arrival circle.
   */
  buryGoal(goal, from) {
    const distance = from ? Math.hypot(goal.x - from.x, goal.y - from.y) : 0;
    const radius = Math.min(DIG_BALL_PX, distance - DIG_CLEARANCE_PX);
    if (radius < DIG_MIN_PX || this.dirtIndex < 0) return false;
    const level = this.world.level;
    const flags = this.engine.materialFlags;
    const reach = Math.ceil(radius);
    for (let dy = -reach; dy <= reach; dy++) {
      const y = Math.round(goal.y) + dy;
      if (y < 0 || y >= level.height) continue;
      for (let dx = -reach; dx <= reach; dx++) {
        const x = Math.round(goal.x) + dx;
        if (x < 0 || x >= level.width || dx * dx + dy * dy > radius * radius) continue;
        const at = y * level.width + x;
        if (flags[level.data[at]] & BACKGROUND) level.data[at] = this.dirtIndex;
      }
    }
    // The whole-level picture is cached between re-reads; this changed it.
    this.mapAt = -1;
    return true;
  }

  /**
   * Hand every worm a destination, if this run has any. Called at the start of
   * an episode and after one resolves, until that worm has received its quota.
   */
  assignGoals(only = null) {
    if (!this.makeGoal) return;
    this.progress.forEach((progress, agent) => {
      if (only !== null && only !== agent) return;
      const assigned = this.totals[agent].goalsAssigned ?? 0;
      if (this.goalsPerEpisode > 0 && assigned >= this.goalsPerEpisode) {
        progress.setGoal(null);
        if (this.views[agent]) this.views[agent].goal = null;
        return;
      }
      const position = this.views[agent]?.self?.position ?? null;
      progress.setGoal(this.makeGoal(this, agent) ?? null, position);
      if (progress.goal?.dig && !this.buryGoal(progress.goal, position)) {
        progress.goal.dig = false;
      }
      if (progress.goal) {
        if (progress.goal.dig) {
          this.totals[agent].goalsDigAssigned = (this.totals[agent].goalsDigAssigned ?? 0) + 1;
        }
        this.totals[agent].goalsAssigned = assigned + 1;
        if (progress.goal.detour) {
          this.totals[agent].goalsDetourAssigned =
            (this.totals[agent].goalsDetourAssigned ?? 0) + 1;
        }
        this.totals[agent].goalAssignedPx =
          (this.totals[agent].goalAssignedPx ?? 0) + (progress.goalDirectPx ?? 0);
        if (assigned === 0 && position) {
          this.totals[agent].goalStartX = position.x;
          this.totals[agent].goalStartY = position.y;
          this.totals[agent].goalTargetX = progress.goal.x;
          this.totals[agent].goalTargetY = progress.goal.y;
          this.totals[agent].goalDetour = progress.goal.detour ? 1 : 0;
          this.totals[agent].goalDig = progress.goal.dig ? 1 : 0;
          this.totals[agent].goalClosestShare = 1;
        }
      }
      if (this.views[agent]) this.views[agent].goal = progress.goal;
    });
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
      let normalized = normalizeAction(action);
      // Movement stages hold the trigger shut. Recoil and the knockback from a
      // worm's own explosions move it further than walking does, and while both
      // are on the books there is no way to tell a policy that goes somewhere
      // from one that gets shoved there. Aim is left alone: the rope fires
      // along it, and the rope is one of the things being taught.
      if (this.lockWeapons) {
        normalized = { ...normalized, keys: normalized.keys & ~KEYS.fire, weapon: 0 };
      }
      // One rope message per cooldown. Throwing and letting go are both
      // messages, so this is "having thrown, live with it for a moment" rather
      // than "throw less" — though it is that too. See `ropeCooldown` in the
      // defaults for what this does and does not change.
      if (this.ropeCooldown > 0 && normalized.rope !== ROPE.none) {
        if (this.decisions < this.ropeReadyAt[agent]) {
          normalized = { ...normalized, rope: ROPE.none };
        } else {
          this.ropeReadyAt[agent] = this.decisions + this.ropeCooldown;
        }
      }
      // A throw is kept for a while: during the hold another throw is ignored
      // and the jump key is dropped, so nothing below can let go of it.
      normalized = this.ropeHolds[agent].apply(normalized);
      // Letting go of the rope is a jump press, because that is what it is in
      // the game: the client sends the release on the press of Jump (with no
      // weapon-change modifier held) and the NinjaRope key only ever throws.
      // So a release presses Jump — which also jumps, if the worm is standing,
      // as it does for a person — and a jump pressed while the rope is out
      // lets go of it. Pressed, not held: the engine's own jump fires on the
      // edge too, and a key held across two decisions was pressed once.
      const wasJumping = ((this.lastActions[agent]?.keys ?? 0) & KEYS.jump) !== 0;
      const jumpPressed = (normalized.keys & KEYS.jump) !== 0 && !wasJumping;
      if (normalized.rope === ROPE.release) {
        normalized = { ...normalized, keys: normalized.keys | KEYS.jump };
      } else if (normalized.rope === ROPE.none && jumpPressed && this.views[agent]?.self?.rope) {
        normalized = { ...normalized, rope: ROPE.release };
      }
      const thrown = normalized.rope === ROPE.throw ? 1 : 0;
      this.ropeThrown[agent] = thrown;
      if (thrown) {
        this.totals[agent].ropeThrows = (this.totals[agent].ropeThrows ?? 0) + 1;
      }
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
    if (this.goalProgressFullAt > 0) {
      const gone = Math.min(
        1,
        Math.max(0, this.decisions - this.goalProgressStartAt) / this.goalProgressFullAt,
      );
      this.goalProgressScale = 1 - (1 - this.goalProgressFloor) * gone;
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
    // A new destination is a new recurrent task even when the surrounding
    // world stays alive. Tell the policy which lanes must forget the route it
    // was following; otherwise a fast worm carries its old plan into the next
    // goal while it is supposed to be producing useful work for the slow one.
    const taskRestarted = new Array(this.agents).fill(false);
    for (let agent = 0; agent < this.agents; agent++) {
      const worm = this.worms[agent];
      const events = this.events[agent];
      events.died = this.alive[agent] && !worm.u ? 1 : 0;
      this.alive[agent] = Boolean(worm.u);
      const hadGoal = Boolean(this.progress[agent].goal);
      const moved = this.progress[agent].update(
        worm.u ? worm : this.views[agent].self.position ?? worm,
        Boolean(worm.u),
      );
      if (this.makeGoal && !hadGoal) {
        this.totals[agent].goalIdleSteps = (this.totals[agent].goalIdleSteps ?? 0) + 1;
      }
      // How near the first task got, as a share of what it had to cover.
      if ((this.totals[agent].goalsAssigned ?? 0) === 1) {
        const tracked = this.progress[agent];
        this.totals[agent].goalClosestShare = moved.reachedGoal
          ? 0
          : tracked.goal && tracked.goalDirectPx > 0 && tracked.goalBestDistance !== null
            ? Math.min(1, Math.max(0, (tracked.goalBestDistance - tracked.options.goalRadiusPx) / tracked.goalDirectPx))
            : this.totals[agent].goalClosestShare ?? 1;
      }
      // Decisions spent hanging off an attached rope. Read after the views were
      // refreshed, so this is the state the worm is in now rather than the one
      // it acted from. Against `ropeThrows` it says whether a throw turns into
      // anything: many throws and almost no held decisions is a worm that lets
      // go the moment it lands.
      if (this.views[agent]?.self?.rope?.attached) {
        this.totals[agent].ropeHeld = (this.totals[agent].ropeHeld ?? 0) + 1;
      }
      // Arriving clears the goal. Hand out another only while this episode's
      // quota has room; fixed-task runs deliberately stop at one.
      if (moved.reachedGoal) {
        // Straight onto the running total, not through `events`. That buffer is
        // reused across steps and `tallyDamage` clears only the fields it owns,
        // so anything else left in it is added again on every later decision of
        // the episode.
        this.totals[agent].goalsReached = (this.totals[agent].goalsReached ?? 0) + 1;
        this.totals[agent].goalStepsReached =
          (this.totals[agent].goalStepsReached ?? 0) + moved.goalSteps;
        this.totals[agent].goalDirectPx =
          (this.totals[agent].goalDirectPx ?? 0) + moved.goalDirectPx;
        this.totals[agent].goalPathPx =
          (this.totals[agent].goalPathPx ?? 0) + moved.goalPathPx;
        this.curriculum?.record(true);
        this.deadlineCurriculum?.record(true);
        this.assignGoals(agent);
        taskRestarted[agent] = Boolean(this.progress[agent].goal);
      } else if (
        this.goalDeadline() > 0 &&
        this.progress[agent].goal &&
        moved.goalSteps >= this.goalDeadline()
      ) {
        // Kept this long and not reached: give it up and hand out another, so
        // a destination the worm cannot get to does not eat the whole episode.
        // Counted, because many of these is the finding, not the reaching.
        this.totals[agent].goalsMissed = (this.totals[agent].goalsMissed ?? 0) + 1;
        this.curriculum?.record(false);
        this.deadlineCurriculum?.record(false);
        this.assignGoals(agent);
        taskRestarted[agent] = Boolean(this.progress[agent].goal);
      }
      const rewardedProgress =
        this.goalProgressMode === "best"
          ? { ...moved, goalDelta: moved.goalBestDelta }
          : moved;
      const outcome = this.reward(
        events,
        { ...rewardedProgress, ...this.#aimAt(agent), ropeThrows: this.ropeThrown[agent] },
        this.weights,
        this.shaping,
        this.goalProgressScale,
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
    const restarted = respawned.map((one, agent) => one || taskRestarted[agent]);
    this.encodeObservations();
    // A task quota is a real episode boundary: once every worm has either
    // arrived or used its deadline, there is no future goal reward to value.
    // The ordinary clock remains a truncation of a world that could continue.
    const goalsResolved =
      this.endOnGoals &&
      this.goalsPerEpisode > 0 &&
      this.totals.every((one) => {
        const assigned = one.goalsAssigned ?? 0;
        const resolved = (one.goalsReached ?? 0) + (one.goalsMissed ?? 0);
        return assigned >= this.goalsPerEpisode && resolved >= assigned;
      });
    const clockEnded = this.world.qb - this.episodeStartTick >= this.episodeTicks;
    this.terminated = goalsResolved;
    this.done = goalsResolved || clockEnded || (this.terminateOnKill && killed);
    this.ending = this.done;
    return {
      observations: this.observations,
      rewards,
      done: this.done,
      // The episode is over and this is its final state; `reset()` has not run.
      terminated: this.terminated,
      truncated: this.done && !this.terminated,
      respawned,
      // Per-policy-lane memory boundary. This includes physical respawns and
      // immediate movement-task turnover while preserving `respawned` for
      // callers that need the literal game event.
      restarted,
      info: { ...this.info(), events: this.events, parts },
    };
  }

  /** Every agent's view of the world as it stands. Cheap: no terrain is copied. */
  refreshViews() {
    this.views = this.worms.map((worm, agent) => {
      const view = viewFromWorld(
        this.world,
        worm,
        this.worms.filter((_, other) => other !== agent),
        this.latency[agent] ?? 0,
        this.lastActions[agent] ?? null,
        this.ropeHolds?.[agent]?.share ?? 0,
      );
      // Where this worm was told to go. `viewFromWorld` only knows what the
      // engine holds, and a goal is this environment's idea, so it is attached
      // here — the vector reads it off the view like everything else.
      view.goal = this.progress[agent]?.goal ?? null;
      return view;
    });
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
      goalProgressScale: this.goalProgressScale,
      goalRadiusPx: this.goalRadius(),
      goalPatience: this.goalDeadline(),
      totals: this.totals,
    };
  }
}
