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
import { MAP_SIZE, WEAPON_SLOTS } from "./observation.js";
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
  "stuckSteps",
  "cellsVisited",
  "fromDamageDealt",
  "fromDamageTaken",
  "fromKill",
  "fromDeath",
  "fromExplore",
  "fromRevisit",
  "fromStuck",
  "fromApproach",
  "fromOnTarget",
  "fromAimedShot",
  "fromGoal",
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
  stuckSteps: "stuckSteps",
  cellsVisited: "cellsVisited",
  fromDamageDealt: "fromDamageDealt",
  fromDamageTaken: "fromDamageTaken",
  fromKill: "fromKill",
  fromDeath: "fromDeath",
  fromExplore: "fromExplore",
  fromRevisit: "fromRevisit",
  fromStuck: "fromStuck",
  fromApproach: "fromApproach",
  fromOnTarget: "fromOnTarget",
  fromAimedShot: "fromAimedShot",
  fromGoal: "fromGoal",
};

export const HEADS = ACTION_HEADS.length;

/**
 * What the `dones` byte says about the observation sent beside it.
 *
 * Three states rather than two, because an episode here ends on a clock this
 * project set and not on anything the game did. Whoever is learning from this
 * has to be able to tell "the match is over, this state is worth nothing" from
 * "we stopped watching, this state is worth whatever it was worth" — and only
 * the second one ever happens here. So the last observation of an episode is
 * sent and marked, and the world restarts on the next call.
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
};

export class VecWormEnv {
  constructor(
    engine,
    { envs = 8, levelPool = 16, levelFiles = [], seed = 1, stagger = false, opponents = 0, ...options } = {},
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
    const level = (_engine, episodeSeed) =>
      this.levels[episodeSeed % this.levels.length];
    this.envs = Array.from(
      { length: envs },
      (_, index) =>
        new WormEnv(engine, {
          ...options,
          // Every world gets its own stream of episodes, so a row of them is a
          // row of different fights and not the same one N times.
          seed: (seed + index * 0x85ebca6b) >>> 0,
          level,
          // The byte forms are what go on the wire; expanding them into planes
          // is the trainer's job, where it is free.
          observations: options.observations ?? ["vector", "patchBytes", "map"],
        }),
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
    this.maps = new Uint8Array(slots * MAP_SIZE);
    this.rewards = new Float32Array(slots);
    this.dones = new Uint8Array(envs);
    // Per worm, not per match: which ones came back from the dead this step.
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
        this.restarts[index * this.agents + agent] = out.respawned[agent] ? 1 : 0;
      }
      if (out.truncated) {
        if (this.warming[index]) {
          this.warming[index] = 0;
          this.dones[index] = DONE.cut;
          continue;
        }
        // Written while the totals are still this episode's; the reset that
        // clears them does not run until the next call.
        this.writeStats(index, env);
        this.dones[index] = DONE.last;
        this.episodes++;
      }
    }
    return this;
  }

  writeStats(index, env) {
    const totals = env.info().totals;
    const mean = (field) =>
      totals.reduce((sum, one) => sum + (one[field] ?? 0), 0) / totals.length;
    // The trained worms are the front of the match and the older copies the
    // back, which is how the trainer seats them.
    const split = totals.length - this.opponents;
    const meanOf = (from, to, field) => {
      if (to <= from) return 0;
      let sum = 0;
      for (let one = from; one < to; one++) sum += totals[one][field] ?? 0;
      return sum / (to - from);
    };
    const versus = (field) =>
      this.opponents
        ? meanOf(0, split, field) - meanOf(split, totals.length, field)
        : 0;
    const at = index * EPISODE_STATS.length;
    for (const [offset, field] of EPISODE_STATS.entries()) {
      if (field === "steps") this.stats[at + offset] = env.episodeTicks / env.frameskip;
      else if (field === "shaping") this.stats[at + offset] = env.shaping;
      else if (field === "seed") this.stats[at + offset] = env.episodeSeed;
      else if (field === "killsVsPast") this.stats[at + offset] = versus("killed");
      else if (field === "damageVsPast") this.stats[at + offset] = versus("damageDealt");
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
      patchCells: this.wantsPatch ? this.spec.patch.cells : 0,
      patchShape: this.wantsPatch ? this.spec.patch.shape : null,
      patchScale: this.spec.patch.scalePx,
      mapCells: this.wantsMap ? MAP_SIZE : 0,
      mapShape: this.wantsMap ? [4, 32, 32] : null,
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
