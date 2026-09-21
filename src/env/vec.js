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
import { MAP_SIZE, PATCH_CELLS, PATCH_SHAPE } from "./observation.js";
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
  "fromGoal",
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
  fromGoal: "fromGoal",
};

export const HEADS = ACTION_HEADS.length;

export class VecWormEnv {
  constructor(
    engine,
    { envs = 8, levelPool = 16, levelFiles = [], seed = 1, ...options } = {},
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
    this.spec = this.envs[0].spec;
    const slots = envs * this.agents;
    this.vectors = new Float32Array(slots * this.spec.vectorSize);
    this.patches = new Uint8Array(slots * PATCH_CELLS);
    this.maps = new Uint8Array(slots * MAP_SIZE);
    this.rewards = new Float32Array(slots);
    this.dones = new Uint8Array(envs);
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
            slot * PATCH_CELLS,
            (slot + 1) * PATCH_CELLS,
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
    this.stats.fill(0);
    for (const env of this.envs) env.reset();
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
    this.dones.fill(0);
    for (let index = 0; index < this.count; index++) {
      const env = this.envs[index];
      for (let agent = 0; agent < this.agents; agent++) {
        const at = (index * this.agents + agent) * HEADS;
        Object.assign(this.actions[agent], actionFromHeads(heads, at));
      }
      const out = env.step(this.actions);
      for (let agent = 0; agent < this.agents; agent++) {
        this.rewards[index * this.agents + agent] = out.rewards[agent];
      }
      if (out.done) {
        this.writeStats(index, env);
        this.dones[index] = 1;
        this.episodes++;
        env.reset();
      }
    }
    return this;
  }

  writeStats(index, env) {
    const totals = env.info().totals;
    const mean = (field) =>
      totals.reduce((sum, one) => sum + (one[field] ?? 0), 0) / totals.length;
    const at = index * EPISODE_STATS.length;
    for (const [offset, field] of EPISODE_STATS.entries()) {
      if (field === "steps") this.stats[at + offset] = env.episodeTicks / env.frameskip;
      else if (field === "seed") this.stats[at + offset] = env.episodeSeed;
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
      weaponIdsCount: 1 + this.spec.foeSlots,
      weaponCount: this.engine.settings.O.length,
      patchCells: this.wantsPatch ? PATCH_CELLS : 0,
      patchShape: this.wantsPatch ? PATCH_SHAPE : null,
      mapCells: this.wantsMap ? MAP_SIZE : 0,
      mapShape: this.wantsMap ? [4, 32, 32] : null,
      statFields: EPISODE_STATS,
      frameskip: this.envs[0].frameskip,
      episodeTicks: this.envs[0].episodeTicks,
      maps: this.levels.length,
      stockMaps: this.stockLevels,
    };
  }
}
