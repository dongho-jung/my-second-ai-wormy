// Run episodes through the headless environment and write down how they went.
//
// The policy here is random: this is the pipeline, not the learner. What it is
// for is (a) proving the environment runs and is reproducible, (b) measuring how
// fast it actually is with observations and rewards in the loop, and (c) giving
// the training monitor something real to draw. A learner writes to exactly the
// same run directory, so the page does not change when one arrives.
import { parseArgs } from "node:util";
import { loadEngine } from "../src/env/engine.js";
import { makeRng } from "../src/env/engine.js";
import { KEYS, ROPE } from "../src/env/actions.js";
import { DEFAULTS, WormEnv } from "../src/env/env.js";
import { observe } from "../src/env/observation.js";
import { PATCH_SIZE } from "../src/env/observation.js";
import { createRun } from "../src/train/recorder.js";

const HELP = `Wormy II — headless environment rollout

Usage: npm run rollout -- [options]

Runs episodes of the headless WebLiero environment with a random policy and
records one line per episode under artifacts/runs/, which "npm run monitor"
draws live.

  --agents 3           Worms in the free-for-all (1 is solo, 2 a duel, 5 a brawl)
  --observation-foes N Size the vector for this many other worms instead of
                       agents-1, so one policy can play any count
  --episodes 20        Episodes to run
  --seed 1             First episode seed; each episode advances it
  --frameskip 4        World ticks per agent step
  --episode-ticks 3600 Ticks per episode (3600 = one minute of game time)
  --latency 0          Input latency in ticks, or a range like 0-3
  --policy random      random or idle
  --observations both  both, vector or patch: the terrain patch costs about ten
                       times the vector, so leaving it out is the fast path
  --level-pool 0       Generate this many levels up front and cycle them; 0 makes
                       a fresh one per episode, which costs about 7 ms each
  --label TEXT         A name for this run on the monitor page
  --no-record          Print only; write nothing to artifacts/runs/
  --quiet              Do not print a line per episode`;

const { values } = parseArgs({
  options: {
    help: { type: "boolean", short: "h" },
    agents: { type: "string", default: "3" },
    "observation-foes": { type: "string" },
    episodes: { type: "string", default: "20" },
    seed: { type: "string", default: "1" },
    frameskip: { type: "string", default: String(DEFAULTS.frameskip) },
    "episode-ticks": { type: "string", default: String(DEFAULTS.episodeTicks) },
    latency: { type: "string", default: "0" },
    policy: { type: "string", default: "random" },
    observations: { type: "string", default: "both" },
    "level-pool": { type: "string", default: "0" },
    label: { type: "string" },
    record: { type: "boolean", default: true },
    quiet: { type: "boolean", default: false },
  },
  allowNegative: true,
});
if (values.help) {
  console.log(HELP);
  process.exit(0);
}
const number = (name, value, { min = 1 } = {}) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < min) {
    throw new Error(`--${name} must be a number of at least ${min}, got ${value}`);
  }
  return parsed;
};
const latency = values.latency.includes("-")
  ? values.latency.split("-").map((part) => number("latency", part, { min: 0 }))
  : number("latency", values.latency, { min: 0 });
if (!["random", "idle"].includes(values.policy)) {
  throw new Error(`--policy must be random or idle, got ${values.policy}`);
}
const KINDS = { both: ["vector", "patch"], vector: ["vector"], patch: ["patch"] };
const observations = KINDS[values.observations];
if (!observations) {
  throw new Error(`--observations must be one of ${Object.keys(KINDS).join(", ")}`);
}

const agents = number("agents", values.agents);
const episodes = number("episodes", values.episodes);
const frameskip = number("frameskip", values.frameskip);
const episodeTicks = number("episode-ticks", values["episode-ticks"]);
const firstSeed = number("seed", values.seed, { min: 0 });

const engine = await loadEngine();
// Generating a level is ~7 ms, which is a fifth of a short episode. A pool keeps
// the variety and pays for it once.
const poolSize = number("level-pool", values["level-pool"], { min: 0 });
const pool = Array.from({ length: poolSize }, (_, index) =>
  engine.randomLevel(firstSeed + index),
);
const env = new WormEnv(engine, {
  agents,
  ...(values["observation-foes"] === undefined
    ? {}
    : { observationFoes: number("observation-foes", values["observation-foes"], { min: 0 }) }),
  frameskip,
  episodeTicks,
  inputLatencyTicks: latency,
  observations,
  seed: firstSeed,
  ...(poolSize
    ? { level: (_engine, seed) => pool[seed % poolSize] }
    : {}),
});

// A policy is a function from an observation to an action. This one ignores the
// observation, which is the point: it is the floor every learner has to beat.
const roll = makeRng(0xc0ffee);
const policies = {
  idle: () => 0,
  random: () => ({
    keys:
      (roll() < 0.5 ? KEYS.left : KEYS.right) |
      (roll() < 0.4 ? KEYS.aimUp : 0) |
      (roll() < 0.2 ? KEYS.aimDown : 0) |
      (roll() < 0.25 ? KEYS.fire : 0) |
      (roll() < 0.08 ? KEYS.jump : 0) |
      (roll() < 0.05 ? KEYS.dig : 0),
    rope: roll() < 0.01 ? ROPE.throw : roll() < 0.02 ? ROPE.release : ROPE.none,
    weapon: roll() < 0.03 ? 1 : 0,
  }),
};
const policy = policies[values.policy];

const run = values.record
  ? await createRun({
      label: values.label ?? `${values.policy} rollout`,
      meta: {
        policy: `${values.policy} policy (no learning)`,
        engineSha256: engine.sha256,
        mod: engine.settings.name,
        agents: env.agents,
        frameskip,
        weights: env.weights,
        episodeTicks,
        inputLatencyTicks: env.inputLatencyTicks,
        loadout: env.loadouts[0].map((id) => engine.weaponNames[id]).join(", "),
        levels: poolSize ? `${poolSize} generated, cycled` : "one generated per episode",
        observation: observations
          .map((kind) =>
            kind === "vector" ? `vector ${env.spec.vectorSize}` : `patch ${PATCH_SIZE}`,
          )
          .join(" + "),
        node: process.version,
      },
    })
  : null;
if (run) {
  console.log(`run ${run.id} -> ${run.path.pathname}`);
  await run.note(`${episodes} episodes with the ${values.policy} policy`);
}

let steps = 0;
let ticks = 0;
const started = performance.now();
for (let episode = 0; episode < episodes; episode++) {
  const { info } = env.reset();
  const episodeStarted = performance.now();
  let episodeSteps = 0;
  let reward = 0;
  let done = false;
  while (!done) {
    const out = env.step(env.observations.map(policy));
    episodeSteps++;
    done = out.done;
  }
  steps += episodeSteps * env.agents;
  ticks += episodeSteps * frameskip;
  // Averaged across the worms: in a free-for-all they are all the same policy,
  // and one worm's good episode is another's bad one.
  const all = env.info().totals;
  const mean = (field) =>
    all.reduce((sum, one) => sum + (one[field] ?? 0), 0) / all.length;
  reward = mean("reward");
  const wall = (performance.now() - episodeStarted) / 1000;
  const damageTaken = mean("damageTaken");
  const line = {
    step: steps,
    episode: env.episode,
    seed: info.seed,
    map: env.world.level.name,
    episodeSteps,
    episodeReward: reward,
    damageDealt: mean("damageDealt"),
    damageTaken,
    selfDamage: mean("selfDamage"),
    damageRatio: damageTaken > 0 ? mean("damageDealt") / damageTaken : 0,
    kills: mean("killed"),
    deaths: mean("died"),
    stuckSteps: mean("stuckSteps"),
    cellsVisited: mean("cellsVisited"),
    fromDamageDealt: mean("fromDamageDealt"),
    fromDamageTaken: mean("fromDamageTaken"),
    fromKill: mean("fromKill"),
    fromDeath: mean("fromDeath"),
    fromExplore: mean("fromExplore"),
    fromRevisit: mean("fromRevisit"),
    fromStuck: mean("fromStuck"),
    stepsPerSecond: (episodeSteps * env.agents) / wall,
    ticksPerSecond: (episodeSteps * frameskip) / wall,
    elapsedSeconds: (performance.now() - started) / 1000,
  };
  await run?.record(line);
  if (!values.quiet) {
    console.log(
      `episode ${String(env.episode).padStart(3)} seed ${String(info.seed).padStart(10)} ` +
        `reward ${reward.toFixed(3).padStart(7)} dealt ${line.damageDealt.toFixed(0).padStart(4)} ` +
        `taken ${line.damageTaken.toFixed(0).padStart(4)} (self ${line.selfDamage.toFixed(0).padStart(4)}) ` +
        `k/d ${line.kills.toFixed(1)}/${line.deaths.toFixed(1)} stuck ${line.stuckSteps.toFixed(0).padStart(3)} ` +
        `${Math.round(line.stepsPerSecond).toLocaleString().padStart(8)} steps/s`,
    );
  }
}
const wall = (performance.now() - started) / 1000;

// What one observation costs, on its own: the environment is the engine plus
// this, and it is worth knowing which half the time went to.
const view = env.views[0];
const into = {
  vector: new Float32Array(env.spec.vectorSize),
  patch: new Float32Array(PATCH_SIZE),
};
const samples = 20_000;
const cost = (kinds) => {
  for (let warm = 0; warm < 2000; warm++) observe(view, into, kinds, env.spec);
  const at = performance.now();
  for (let sample = 0; sample < samples; sample++) observe(view, into, kinds, env.spec);
  return (performance.now() - at) / samples;
};
const observeMs = cost(observations);
const vectorMs = cost(["vector"]);

const summary = {
  episodes,
  agentSteps: steps,
  worldTicks: ticks,
  seconds: Number(wall.toFixed(2)),
  stepsPerSecond: Math.round(steps / wall),
  ticksPerSecond: Math.round(ticks / wall),
  realtimeFactor: Math.round(ticks / wall / 60),
  observeMs: Number(observeMs.toFixed(4)),
  vectorOnlyMs: Number(vectorMs.toFixed(4)),
};
console.log(
  `\n${summary.episodes} episodes: ${summary.agentSteps.toLocaleString()} agent steps, ` +
    `${summary.worldTicks.toLocaleString()} world ticks in ${summary.seconds}s\n` +
    `  ${summary.stepsPerSecond.toLocaleString()} agent steps/s | ` +
    `${summary.ticksPerSecond.toLocaleString()} ticks/s = ${summary.realtimeFactor.toLocaleString()}x realtime\n` +
    `  one observation (${observations.join(" + ")}): ${summary.observeMs} ms` +
    ` | vector alone: ${summary.vectorOnlyMs} ms`,
);
if (run) {
  await run.note(
    `finished: ${summary.stepsPerSecond.toLocaleString()} steps/s, ` +
      `${summary.observeMs} ms per observation`,
  );
  await run.close({ status: "done", ...summary });
}
