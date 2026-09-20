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
import { PATCH_CELLS } from "./observation.js";
import { WormEnv } from "./env.js";

/** Per-episode numbers, in the order they are written. Averaged over the worms. */
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
  "seed",
];

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
      return engine.readLevel(path.split("/").pop().replace(/\.lev$/i, ""), bytes);
    });
    this.levels = [
      ...Array.from({ length: Math.max(1, levelPool) }, (_, index) =>
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
          // The byte patch is what goes on the wire; the one-hot expansion is
          // the trainer's job, where it is free.
          observations: options.observations ?? ["vector", "patchBytes"],
        }),
    );
    this.agents = this.envs[0].agents;
    this.spec = this.envs[0].spec;
    const slots = envs * this.agents;
    this.vectors = new Float32Array(slots * this.spec.vectorSize);
    this.patches = new Uint8Array(slots * PATCH_CELLS);
    this.rewards = new Float32Array(slots);
    this.dones = new Uint8Array(envs);
    this.stats = new Float32Array(envs * EPISODE_STATS.length);
    this.wantsPatch = this.envs[0].observationKinds.includes("patchBytes");

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
    this.stats[at] = env.elapsedSteps ?? env.episodeTicks / env.frameskip;
    this.stats[at + 1] = mean("reward");
    this.stats[at + 2] = mean("damageDealt");
    this.stats[at + 3] = mean("damageTaken");
    this.stats[at + 4] = mean("selfDamage");
    this.stats[at + 5] = mean("killed");
    this.stats[at + 6] = mean("died");
    this.stats[at + 7] = mean("stuckSteps");
    this.stats[at + 8] = mean("cellsVisited");
    this.stats[at + 9] = env.episodeSeed;
  }

  /** What a consumer needs to know to read the buffers. */
  describe() {
    return {
      envs: this.count,
      agents: this.agents,
      heads: ACTION_HEADS.map(([name, choices]) => ({ name, choices: choices.length })),
      vectorSize: this.spec.vectorSize,
      patchCells: this.wantsPatch ? PATCH_CELLS : 0,
      patchShape: this.wantsPatch ? [4, 32, 32] : null,
      statFields: EPISODE_STATS,
      frameskip: this.envs[0].frameskip,
      episodeTicks: this.envs[0].episodeTicks,
      maps: this.levels.length,
      stockMaps: this.stockLevels,
    };
  }
}
