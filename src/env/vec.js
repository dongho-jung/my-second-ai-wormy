// Many matches in one process, laid out the way a trainer wants to read them.
//
// A single environment spends most of its time being called rather than
// working, and a policy that decides for one worm at a time wastes the batch
// its network was built for. This runs a row of worlds together and writes
// every observation into one flat buffer per kind, so a whole step crosses to
// the trainer as a handful of memory blocks rather than a few hundred objects.
//
// The worlds share nothing but the engine and a pool of maps. Each has its own
// terrain to dig, its own seed and its own episode clock, and one finishing
// does not interrupt the others: it restarts on the spot and the trainer is
// told where the boundary was.
import { readFileSync } from "node:fs";
import { actionFromHeads, ACTION_HEADS } from "./actions.js";
import { MAP, MAP_SIZE, PATCH, WEAPON_SLOTS } from "./observation.js";
import { WormEnv } from "./env.js";

/**
 * Per-episode numbers, in the order they are written, averaged over the worms.
 *
 * The `from*` half is the reward broken into the terms it is made of. Without
 * it a run reports one number going up or down and no way to tell whether it is
 * winning fights or collecting an exploration bonus for tunnelling — which is
 * the first question anyone asks when the behaviour looks wrong.
 */
export const EPISODE_STATS = [
  "steps",
  "reward",
  "damageDealt",
  "damageTaken",
  "selfDamage",
  "kills",
  "deaths",
  // Of those deaths, the ones nobody else caused.
  "suicides",
  "stuckSteps",
  "cellsVisited",
  // The share of the match spent hanging from an attached rope.
  "ropeShare",
  "fromDamageDealt",
  "fromDamageTaken",
  "fromKill",
  "fromDeath",
  "fromSuicide",
  "fromExplore",
  "fromRevisit",
  "fromStuck",
  "fromApproach",
  "fromOnTarget",
  "fromAimedShot",
  "fromGoal",
  "fromGoalSpeed",
  "fromRopeThrow",
  // How many destinations it actually reached this episode. `fromGoal` mixes
  // arriving with closing distance, so on its own it cannot say whether a worm
  // is getting there or just drifting the right way. This is the number a
  // movement run is judged on.
  "goalsReached",
  // And how many it was given up on: kept longer than the world's patience and
  // swapped for another. Many of these against few reached is a worm that
  // cannot get where it is sent, whatever the reward curve says.
  "goalsMissed",
  // Every assigned destination, including one still unresolved at the episode
  // boundary, and its mean initial straight-line distance. A fixed benchmark
  // checks these beside the seed to prove both policies received the same task.
  "goalsAssigned",
  // Decisions taken with no destination while another worm held the shared
  // episode open. Continuous movement training should keep this at zero.
  "goalIdleSteps",
  "goalAssignedDistance",
  // How many assigned tasks require crossing solid terrain on the direct line,
  // and whether the first task is one of them. Fixed benchmarks use one task,
  // so `goalDetour` splits the scoreboard into open and route-planning cases.
  "goalsDetourAssigned",
  "goalDetour",
  // The first task's exact endpoints. Benchmarks run one worm and one task, so
  // these make equality stronger than "the distances happened to match".
  "goalStartX",
  "goalStartY",
  "goalTargetX",
  "goalTargetY",
  // Speed is measured on completed destinations. Seconds says what a person
  // sees, direct px/s separates it from the random distance of each goal, and
  // path efficiency says how much of the travelled route actually shortened
  // the straight line. Together they distinguish "got lucky with short goals"
  // from "takes a direct route quickly".
  "goalSeconds",
  "goalSpeed",
  "goalPathEfficiency",
  // How far this world was drawing destinations when the match ended. On a
  // clock schedule every world says the same; on a success-driven one each
  // world has moved it as far as its own worms earned.
  "goalRadiusPx",
  // The other half of the speed curriculum: how long a goal currently gets,
  // and how much of the old distance shaping is still paid.
  "goalPatience",
  "goalProgressScale",
  // The rope, which is the one tool a worm has for ground it cannot walk to.
  // Throws are what it asked for; held is what it got — decisions spent with a
  // rope actually attached. A policy near maximum entropy throws on a third of
  // its decisions and lets go on a third, so the two numbers apart say whether
  // any of that turned into hanging off something.
  "ropeThrows",
  "ropeHeld",
  "shaping",
  // The only figure that answers "is it getting better" rather than "is
  // something happening".
  //
  // Everything above is averaged over every worm in the match, and when all of
  // them are the same policy that average goes up whenever the three of them
  // get more reckless together — which looks exactly like improving. These two
  // are the worms being trained minus the older copies they are playing, on the
  // same map, in the same match. Zero when nobody is an older copy, because
  // then there is nothing to be better than.
  "killsVsPast",
  "damageVsPast",
  // The same difference for the other two things a fight is scored on, so a
  // learner that kills more than its past self but also dies more, or hurts
  // itself more, is not read as simply better.
  "deathsVsPast",
  "selfDamageVsPast",
  "suicidesVsPast",
  // Movement counterparts of the combat differences above. They let the
  // evaluator recover each side after swapping seats on identical maps.
  "goalsVsPast",
  "goalsMissedVsPast",
  "goalSecondsVsPast",
  "goalSpeedVsPast",
  "goalEfficiencyVsPast",
  // What the match is actually scored on — damage, kills and deaths, with no
  // ladder in it — averaged over the worms being trained only. The best
  // checkpoint is picked on this rather than on `reward`: with the ladder
  // fading over a run, the total reward of a later, better policy reads lower
  // than an earlier one's, and "best" would freeze halfway through.
  "combat",
  // Small, exact index into this worker's configured level list. Fixed exams
  // retain it beside the endpoints so "same seed" is not the only map proof.
  "mapIndex",
  // Float32 cannot hold every uint32 seed exactly. Split halves retain the
  // replay key while the legacy combined field remains available.
  "seedLow",
  "seedHigh",
  "seed",
];

// What each of them is called in a worm's running totals. Written by name so
// adding one cannot silently shift every number after it.
const STAT_SOURCE = {
  reward: "reward",
  damageDealt: "damageDealt",
  damageTaken: "damageTaken",
  selfDamage: "selfDamage",
  kills: "killed",
  deaths: "died",
  suicides: "suicides",
  stuckSteps: "stuckSteps",
  cellsVisited: "cellsVisited",
  fromDamageDealt: "fromDamageDealt",
  fromDamageTaken: "fromDamageTaken",
  fromKill: "fromKill",
  fromDeath: "fromDeath",
  fromSuicide: "fromSuicide",
  fromExplore: "fromExplore",
  fromRevisit: "fromRevisit",
  fromStuck: "fromStuck",
  fromApproach: "fromApproach",
  fromOnTarget: "fromOnTarget",
  fromAimedShot: "fromAimedShot",
  fromGoal: "fromGoal",
  fromGoalSpeed: "fromGoalSpeed",
  fromRopeThrow: "fromRopeThrow",
  goalsReached: "goalsReached",
  goalsMissed: "goalsMissed",
  goalsAssigned: "goalsAssigned",
  goalIdleSteps: "goalIdleSteps",
  goalAssignedDistance: "goalAssignedDistance",
  goalsDetourAssigned: "goalsDetourAssigned",
  goalDetour: "goalDetour",
  goalStartX: "goalStartX",
  goalStartY: "goalStartY",
  goalTargetX: "goalTargetX",
  goalTargetY: "goalTargetY",
  goalSeconds: "goalSeconds",
  goalSpeed: "goalSpeed",
  goalPathEfficiency: "goalPathEfficiency",
  ropeThrows: "ropeThrows",
  ropeHeld: "ropeHeld",
};

export const HEADS = ACTION_HEADS.length;

/**
 * What the `dones` byte says about the observation sent beside it.
 *
 * More than a boolean because a fixed clock and a completed task mean
 * different things to the value function. The last observation is always sent
 * and marked, and the world restarts on the next call.
 */
export const DONE = {
  /** Mid-episode. The usual. */
  ongoing: 0,
  /** The first observation of a new episode; whatever came before it is gone. */
  first: 1,
  /** The last observation of an episode. Value it; do not treat it as an end. */
  last: 2,
  // The episode was cut short on purpose — the staggered first one every
  // world plays — so its last observation is still valued like a truncation,
  // but nothing is filed about it: it was not a whole episode.
  cut: 3,
  /** A complete task. Its closing observation has no future value. */
  terminal: 4,
};

export class VecWormEnv {
  constructor(
    engine,
    {
      envs = 8,
      levelPool = 16,
      levelFiles = [],
      levelSequence = "seed",
      levelOffset = 0,
      levelStride = null,
      seed = 1,
      stagger = false,
      opponents = 0,
      ...options
    } = {},
  ) {
    if (!Number.isInteger(envs) || envs < 1) {
      throw new Error(`envs must be a whole number of at least 1, got ${envs}`);
    }
    this.engine = engine;
    this.count = envs;
    // Two kinds of map, because a real room has two kinds. A host who asks for
    // a random one gets the generated dirt; a host who picks a level gets one
    // of the stock files, which are almost entirely rock. Train on only the
    // first and the policy has never seen a map it cannot dig through.
    //
    // Generating one costs about seven milliseconds, a fifth of a short
    // episode, so the pool is built once and cycled.
    const stock = levelFiles.map((path) => {
      const bytes = readFileSync(path);
      const file = path.split("/").pop();
      // The name keeps its extension: it decides which reader the file gets,
      // and the community pools ship PNGs beside the game's own .lev files.
      return engine.readAnyLevel(file, bytes);
    });
    // Generated maps can be turned off outright, but only when there is
    // something else to play on: a run with no levels at all has nowhere to put
    // a worm. Asking for none and silently getting one is how a run that was
    // meant to be on the room's own maps spends a twentieth of its episodes on
    // a dirt field instead.
    const generated = stock.length ? Math.max(0, levelPool) : Math.max(1, levelPool);
    this.levels = [
      ...Array.from({ length: generated }, (_, index) =>
        engine.randomLevel((seed + index * 0x9e3779b9) >>> 0, options.levelOptions),
      ),
      ...stock,
    ];
    this.stockLevels = stock.length;
    if (!["seed", "roundRobin"].includes(levelSequence)) {
      throw new Error(`levelSequence must be seed or roundRobin, got ${levelSequence}`);
    }
    const firstLevel = Math.trunc(levelOffset);
    const stride = Math.max(1, Math.trunc(levelStride ?? envs));
    this.envs = Array.from(
      { length: envs },
      (_, index) => {
        let episode = 0;
        let environment = null;
        const level = (_engine, episodeSeed) => {
          let at = episodeSeed % this.levels.length;
          if (levelSequence === "roundRobin") {
            const scheduled = firstLevel + index + episode++ * stride;
            at = ((scheduled % this.levels.length) + this.levels.length) % this.levels.length;
          }
          if (environment) environment.levelIndex = at;
          return this.levels[at];
        };
        environment = new WormEnv(engine, {
          ...options,
          // Every world gets its own stream of episodes, so a row of them is a
          // row of different fights and not the same one N times.
          seed: (seed + index * 0x85ebca6b) >>> 0,
          level,
          // The byte forms are what go on the wire; expanding them into planes
          // is the trainer's job, where it is free.
          // A second patch scale means a second cut in every observation,
          // whether or not the caller listed the kinds itself: a checkpoint's
          // saved world names the kinds it trained with, and the second cut
          // is only ever added on top of those.
          observations: [
            ...(options.observations ?? ["vector", "patchBytes", "map"]).filter(
              (kind) => kind !== "patchBytes2",
            ),
            ...(options.patchScale2 ? ["patchBytes2"] : []),
          ],
        });
        return environment;
      },
    );
    this.agents = this.envs[0].agents;
    // How many worms at the end of each match are older copies of the policy
    // rather than the one being trained. The trainer decides this and drives
    // them; all this side needs is where they sit, so that "us" and "them" can
    // be reported apart instead of averaged into one number.
    this.opponents = Math.max(0, Math.min(this.agents - 1, Math.trunc(opponents)));
    this.spec = this.envs[0].spec;
    const slots = envs * this.agents;
    this.vectors = new Float32Array(slots * this.spec.vectorSize);
    this.patches = new Uint8Array(slots * this.spec.patch.cells);
    this.wantsPatch2 = Boolean(this.spec.patch2) && this.envs[0].observationKinds.includes("patchBytes2");
    this.patches2 = new Uint8Array(this.wantsPatch2 ? slots * this.spec.patch2.cells : 0);
    this.maps = new Uint8Array(slots * MAP_SIZE);
    this.rewards = new Float32Array(slots);
    this.dones = new Uint8Array(envs);
    // Per worm, not per match: which policy lanes crossed a memory boundary
    // this step, either by respawning or by receiving a fresh movement task.
    this.restarts = new Uint8Array(slots);
    // Which worlds are still on their cut-short first episode.
    this.warming = new Uint8Array(envs);
    this.stagger = Boolean(stagger);
    this.stats = new Float32Array(envs * EPISODE_STATS.length);
    this.wantsPatch = this.envs[0].observationKinds.includes("patchBytes");
    this.wantsMap = this.envs[0].observationKinds.includes("map");

    // Hand each world a window onto the flat buffers as its own scratch, so an
    // observation is encoded straight into the block that will be sent and
    // never copied again.
    this.envs.forEach((env, index) => {
      env.observations = Array.from({ length: this.agents }, (_, agent) => {
        const slot = index * this.agents + agent;
        const into = {
          vector: this.vectors.subarray(
            slot * this.spec.vectorSize,
            (slot + 1) * this.spec.vectorSize,
          ),
        };
        if (this.wantsPatch) {
          into.patchBytes = this.patches.subarray(
            slot * this.spec.patch.cells,
            (slot + 1) * this.spec.patch.cells,
          );
        }
        if (this.wantsPatch2) {
          into.patchBytes2 = this.patches2.subarray(
            slot * this.spec.patch2.cells,
            (slot + 1) * this.spec.patch2.cells,
          );
        }
        if (this.wantsMap) {
          into.map = this.maps.subarray(slot * MAP_SIZE, (slot + 1) * MAP_SIZE);
        }
        return into;
      });
    });
    this.actions = Array.from({ length: this.agents }, () => ({
      keys: 0,
      rope: 0,
      weapon: 0,
    }));
    this.episodes = 0;
  }

  /** Start every world. Called once; after that they restart themselves. */
  reset() {
    this.rewards.fill(0);
    this.dones.fill(0);
    this.restarts.fill(0);
    this.stats.fill(0);
    this.warming.fill(0);
    this.envs.forEach((env, index) => {
      env.reset();
      if (!this.stagger) return;
      // Start each world's first episode a different way through, spread
      // evenly, so the matches end at different times from then on. Started
      // together they would end together, every episode, and the trainer
      // would see its statistics arrive in one lump every seven updates with
      // nothing in between. The cut-short episode is not filed as one.
      const offset = Math.floor((index / this.count) * env.episodeTicks);
      if (!offset) return; // the first world's first episode is a whole one
      env.episodeStartTick -= offset;
      this.warming[index] = 1;
    });
    return this;
  }

  /**
   * One decision for every worm in every world, as `envs * agents * HEADS`
   * head choices. Worlds that finish are recorded and restarted, and the
   * observation that comes back for them is the first of the new episode.
   */
  step(heads) {
    const expected = this.count * this.agents * HEADS;
    if (heads.length !== expected) {
      throw new Error(`expected ${expected} head choices, got ${heads.length}`);
    }
    this.dones.fill(DONE.ongoing);
    this.restarts.fill(0);
    for (let index = 0; index < this.count; index++) {
      const env = this.envs[index];
      // A world that ended last call starts over now rather than stepping. Its
      // final observation has already been sent and valued, and the action that
      // came back for it has nowhere to go: the match it belonged to is over.
      // One decision per episode is spent this way, out of nine hundred.
      if (env.done) {
        env.reset();
        this.dones[index] = DONE.first;
        for (let agent = 0; agent < this.agents; agent++) {
          this.rewards[index * this.agents + agent] = 0;
        }
        continue;
      }
      for (let agent = 0; agent < this.agents; agent++) {
        const at = (index * this.agents + agent) * HEADS;
        Object.assign(this.actions[agent], actionFromHeads(heads, at));
      }
      const out = env.step(this.actions);
      for (let agent = 0; agent < this.agents; agent++) {
        this.rewards[index * this.agents + agent] = out.rewards[agent];
        const restarted = out.restarted ?? out.respawned;
        this.restarts[index * this.agents + agent] = restarted[agent] ? 1 : 0;
      }
      if (out.done) {
        if (this.warming[index]) {
          this.warming[index] = 0;
          this.dones[index] = DONE.cut;
          continue;
        }
        // Written while the totals are still this episode's; the reset that
        // clears them does not run until the next call.
        this.writeStats(index, env);
        this.dones[index] = out.terminated ? DONE.terminal : DONE.last;
        this.episodes++;
      }
    }
    return this;
  }

  writeStats(index, env) {
    const info = env.info();
    const totals = info.totals;
    const episodeSteps = (info.elapsedTicks ?? env.episodeTicks) / env.frameskip;
    const valueOf = (one, field) => {
      const reached = one.goalsReached ?? 0;
      if (field === "goalSeconds") {
        // With no arrival, zero seconds would call the worst policy the
        // fastest. Use the current deadline (or the whole episode when goals
        // never expire) as the honest lower bound on how long it failed for.
        const decisions =
          reached > 0
            ? (one.goalStepsReached ?? 0) / reached
            : env.goalDeadline() || env.episodeTicks / env.frameskip;
        return decisions * env.frameskip / 60;
      }
      if (field === "goalAssignedDistance") {
        const assigned = one.goalsAssigned ?? 0;
        return assigned > 0 ? (one.goalAssignedPx ?? 0) / assigned : 0;
      }
      if (field === "goalSpeed") {
        const decisions = one.goalStepsReached ?? 0;
        return decisions > 0
          ? ((one.goalDirectPx ?? 0) / decisions) * 60 / env.frameskip
          : 0;
      }
      if (field === "goalPathEfficiency") {
        const path = one.goalPathPx ?? 0;
        return path > 0 ? Math.min(1, (one.goalDirectPx ?? 0) / path) : 0;
      }
      return one[field] ?? 0;
    };
    const mean = (field) =>
      totals.reduce((sum, one) => sum + valueOf(one, field), 0) / totals.length;
    // The trained worms are the front of the match and the older copies the
    // back, which is how the trainer seats them.
    const split = totals.length - this.opponents;
    const meanOf = (from, to, field) => {
      if (to <= from) return 0;
      let sum = 0;
      for (let one = from; one < to; one++) sum += valueOf(totals[one], field);
      return sum / (to - from);
    };
    const versus = (field) =>
      this.opponents
        ? meanOf(0, split, field) - meanOf(split, totals.length, field)
        : 0;
    const at = index * EPISODE_STATS.length;
    for (const [offset, field] of EPISODE_STATS.entries()) {
      if (field === "steps") this.stats[at + offset] = episodeSteps;
      else if (field === "ropeShare") {
        this.stats[at + offset] = mean("ropeHeld") / Math.max(1, episodeSteps);
      }
      else if (field === "shaping") this.stats[at + offset] = env.shaping;
      else if (field === "goalRadiusPx") this.stats[at + offset] = env.goalRadius() ?? 0;
      else if (field === "goalPatience") this.stats[at + offset] = env.goalDeadline();
      else if (field === "goalProgressScale") this.stats[at + offset] = env.goalProgressScale;
      else if (field === "mapIndex") {
        this.stats[at + offset] = env.levelIndex ?? env.episodeSeed % this.levels.length;
      }
      else if (field === "seedLow") this.stats[at + offset] = env.episodeSeed & 0xffff;
      else if (field === "seedHigh") this.stats[at + offset] = env.episodeSeed >>> 16;
      else if (field === "seed") this.stats[at + offset] = env.episodeSeed;
      else if (field === "killsVsPast") this.stats[at + offset] = versus("killed");
      else if (field === "damageVsPast") this.stats[at + offset] = versus("damageDealt");
      else if (field === "deathsVsPast") this.stats[at + offset] = versus("died");
      else if (field === "selfDamageVsPast") this.stats[at + offset] = versus("selfDamage");
      else if (field === "suicidesVsPast") this.stats[at + offset] = versus("suicides");
      else if (field === "goalsVsPast") this.stats[at + offset] = versus("goalsReached");
      else if (field === "goalsMissedVsPast") this.stats[at + offset] = versus("goalsMissed");
      else if (field === "goalSecondsVsPast") this.stats[at + offset] = versus("goalSeconds");
      else if (field === "goalSpeedVsPast") this.stats[at + offset] = versus("goalSpeed");
      else if (field === "goalEfficiencyVsPast") {
        this.stats[at + offset] = versus("goalPathEfficiency");
      }
      else if (field === "combat") {
        this.stats[at + offset] =
          meanOf(0, split, "fromDamageDealt") +
          meanOf(0, split, "fromDamageTaken") +
          meanOf(0, split, "fromKill") +
          meanOf(0, split, "fromDeath");
      }
      else this.stats[at + offset] = mean(STAT_SOURCE[field]);
    }
  }

  /** What a consumer needs to know to read the buffers. */
  describe() {
    return {
      envs: this.count,
      agents: this.agents,
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      vectorSize: this.spec.vectorSize,
      // Where the weapon identities sit in the vector, and how many there are
      // to tell apart. The network embeds them; nothing else should normalise
      // or interpolate them.
      weaponIdsAt: this.spec.offsets.weaponIds,
      weaponIdsCount: WEAPON_SLOTS + this.spec.foeSlots,
      weaponCount: this.engine.settings.O.length,
      // Whether the mod's weapons were measured. Without the profile the
      // weapon features above are all zero and the aim rewards pay nothing;
      // a trainer must refuse rather than run blind.
      weaponsMeasured: Boolean(this.engine.weaponsMeasured),
      patchCells: this.wantsPatch ? this.spec.patch.cells : 0,
      patchShape: this.wantsPatch ? this.spec.patch.shape : null,
      patchScale: this.spec.patch.scalePx,
      // What a cell's byte expands to, so a trainer built for another picture
      // refuses rather than reading four kinds of ground as three.
      patchChannels: PATCH.channels.length,
      mapChannels: MAP.channels.length,
      // The second cut, when one was asked for; zero cells otherwise.
      patch2Cells: this.wantsPatch2 ? this.spec.patch2.cells : 0,
      patch2Shape: this.wantsPatch2 ? this.spec.patch2.shape : null,
      patch2Scale: this.wantsPatch2 ? this.spec.patch2.scalePx : null,
      mapCells: this.wantsMap ? MAP_SIZE : 0,
      mapShape: this.wantsMap ? [MAP.channels.length, MAP.cells, MAP.cells] : null,
      statFields: EPISODE_STATS,
      // Spelled out on the wire so the other side cannot drift from it.
      doneCodes: DONE,
      opponents: this.opponents,
      frameskip: this.envs[0].frameskip,
      episodeTicks: this.envs[0].episodeTicks,
      maps: this.levels.length,
      stockMaps: this.stockLevels,
    };
  }
}
