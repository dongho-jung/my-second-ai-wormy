// The training page. It reads a run directory through the monitor and draws how
// the run has gone; it can start and change nothing.
//
// Every numeric field in a record becomes a chart, in the order the run first
// mentioned it, so a trainer that begins logging a new quantity shows up here
// without this file being touched. The names below only decide what a known
// field is called and which direction is the good one.
const element = (id) => document.getElementById(id);

const SERIES = {
  episodeReward: { label: "reward per episode", good: "up" },
  killsVsPast: { label: "kills: learners minus their past selves", good: "up" },
  damageVsPast: { label: "damage dealt: learners minus their past selves", good: "up" },
  deathsVsPast: { label: "deaths: learners minus their past selves", good: "down" },
  selfDamageVsPast: { label: "damage to itself: learners minus their past selves", good: "down" },
  suicidesVsPast: { label: "deaths by its own hand: learners minus their past selves", good: "down" },
  probeKills: { label: "kills a match against worms that do nothing", good: "up" },
  probeDeaths: { label: "deaths a match against worms that do nothing", good: "down" },
  probeSuicides: { label: "deaths by its own hand against worms that do nothing", good: "down" },
  probeDamageDealt: { label: "damage dealt against worms that do nothing", good: "up" },
  probeSelfDamage: { label: "damage to itself against worms that do nothing", good: "down" },
  probeSeconds: { label: "seconds a probe took" },
  combat: { label: "combat score: damage, kills, deaths, learners only", good: "up" },
  bestCombat: { label: "best combat score kept", good: "up" },
  bestGoals: { label: "best destinations a match kept", good: "up" },
  bestBenchmarkSuccess: { label: "best fixed-scenario success", good: "up" },
  bestBenchmarkSeconds: { label: "best fixed-scenario seconds", good: "down" },
  benchmarkSuccess: { label: "fixed-scenario success", good: "up" },
  benchmarkReached: { label: "fixed scenarios reached", good: "up" },
  benchmarkEpisodes: { label: "fixed scenarios tested" },
  benchmarkSeconds: { label: "seconds per fixed scenario, failures included", good: "down" },
  benchmarkSpeed: { label: "fixed-scenario direct speed, px/s", good: "up" },
  benchmarkEfficiency: { label: "fixed-scenario path efficiency", good: "up" },
  benchmarkDistance: { label: "fixed-scenario mean distance, px" },
  benchmarkDetourSuccess: { label: "fixed detour-scenario success", good: "up" },
  benchmarkDetourReached: { label: "fixed detour scenarios reached", good: "up" },
  benchmarkDetourEpisodes: { label: "fixed detour scenarios tested" },
  benchmarkDetourSeconds: { label: "seconds per fixed detour scenario, failures included", good: "down" },
  benchmarkWallSeconds: { label: "seconds the fixed benchmark took" },
  validationSuccess: { label: "validation success, every suite pooled", good: "up" },
  validationSeconds: { label: "validation seconds per route, failures included", good: "down" },
  championRoutes: { label: "routes compared with the champion" },
  championSeconds: { label: "the champion's seconds per route, same suites", good: "down" },
  championDeltaSeconds: { label: "seconds per route against the champion, same routes", good: "down" },
  championDeltaSecondsLow: { label: "seconds against the champion: 95% interval, low end", good: "down" },
  championDeltaSecondsHigh: { label: "seconds against the champion: 95% interval, high end", good: "down" },
  championDeltaSuccess: { label: "success against the champion, same routes", good: "up" },
  championDeltaSuccessLow: { label: "success against the champion: 95% interval, low end", good: "up" },
  championDeltaSuccessHigh: { label: "success against the champion: 95% interval, high end", good: "up" },
  championGained: { label: "routes solved that the champion does not", good: "up" },
  championLost: { label: "routes the champion solves and this does not", good: "down" },
  stabilityStrikes: { label: "checks in a row significantly slower than the champion", good: "down" },
  rollbackCount: { label: "champion restores" },
  testSuccess: { label: "test-suite success (start and end only)", good: "up" },
  testSeconds: { label: "test-suite seconds per route (start and end only)", good: "down" },
  testStartDeltaSeconds: { label: "test seconds per route, selected minus start", good: "down" },
  testStartDeltaSuccess: { label: "test success, selected minus start", good: "up" },
  bestReward: { label: "best reward kept", good: "up" },
  goalsReached: { label: "destinations reached a match", good: "up" },
  goalsMissed: { label: "destinations given up on a match", good: "down" },
  goalsAssigned: { label: "destinations assigned a match" },
  goalAssignedDistance: { label: "mean assigned distance, px" },
  goalSuccess: { label: "resolved destinations reached", good: "up" },
  goalSeconds: { label: "seconds per reached destination", good: "down" },
  goalSpeed: { label: "straight-line speed to destinations, px/s", good: "up" },
  goalPathEfficiency: { label: "straight distance divided by route length", good: "up" },
  goalsVsPast: { label: "destinations: learners minus their past selves", good: "up" },
  goalsMissedVsPast: { label: "missed destinations: learners minus their past selves", good: "down" },
  goalSecondsVsPast: { label: "seconds per destination: learners minus their past selves", good: "down" },
  goalSpeedVsPast: { label: "destination speed: learners minus their past selves", good: "up" },
  goalEfficiencyVsPast: { label: "route efficiency: learners minus their past selves", good: "up" },
  goalRadiusPx: { label: "how far a destination may be, px" },
  goalPatience: { label: "destination deadline, decisions", good: "down" },
  goalProgressScale: { label: "share of per-pixel goal shaping left", good: "down" },
  ropeThrows: { label: "rope throws a match" },
  ropeHeld: { label: "decisions a match spent on the rope" },
  fromRopeThrow: { label: "reward from: throwing the rope", good: "up" },
  entropyMove: { label: "entropy of the move head" },
  entropyAim: { label: "entropy of the aim head" },
  entropyFire: { label: "entropy of the fire head" },
  entropyJump: { label: "entropy of the jump head" },
  entropyDig: { label: "entropy of the dig head" },
  entropyRope: { label: "entropy of the rope head" },
  entropyRopeLength: { label: "entropy of the rope length head" },
  entropyWeapon: { label: "entropy of the weapon head" },
  kills: { label: "kills", good: "up" },
  deaths: { label: "deaths", good: "down" },
  suicides: { label: "deaths by its own hand", good: "down" },
  damageDealt: { label: "damage dealt", good: "up" },
  damageTaken: { label: "damage taken", good: "down" },
  selfDamage: { label: "damage to itself", good: "down" },
  damageRatio: { label: "damage dealt per taken", good: "up" },
  stuckSteps: { label: "steps stuck", good: "down" },
  cellsVisited: { label: "ground covered", good: "up" },
  ropeShare: { label: "share of the match hanging from the rope", good: "up" },
  entropy: { label: "entropy: how undecided it still is" },
  demoAgreement: { label: "agrees with recorded play", good: "up" },
  demoFrames: { label: "frames of recorded play", good: "up" },
  bcLoss: { label: "disagreement with recorded play", good: "down" },
  learningRate: { label: "learning rate" },
  policyLoss: { label: "policy loss", good: "down" },
  valueLoss: { label: "value loss", good: "down" },
  explainedVariance: { label: "explained variance", good: "up" },
  approxKL: { label: "KL per update", good: "down" },
  clipFraction: { label: "clipped fraction" },
  fromDamageDealt: { label: "reward from: damage dealt", good: "up" },
  fromDamageTaken: { label: "reward from: damage taken", good: "up" },
  fromKill: { label: "reward from: kills", good: "up" },
  fromDeath: { label: "reward from: deaths", good: "up" },
  fromSuicide: { label: "reward from: dying by its own hand", good: "up" },
  fromExplore: { label: "reward from: new ground", good: "up" },
  fromRevisit: { label: "reward from: doubling back", good: "up" },
  fromStuck: { label: "reward from: being stuck", good: "up" },
  fromGoal: { label: "reward from: the goal", good: "up" },
  fromGoalSpeed: { label: "reward from: arriving quickly", good: "up" },
  shaping: { label: "the ladder's weight", good: "down" },
  fromApproach: { label: "reward from: closing on somebody", good: "up" },
  fromOnTarget: { label: "reward from: aiming at somebody", good: "up" },
  fromAimedShot: { label: "reward from: firing while aimed", good: "up" },
  reward: { label: "reward per step", good: "up" },
  meanReward: { label: "reward per rollout", good: "up" },
  episodeSteps: { label: "episode length in steps" },
  episodes: { label: "episodes finished" },
  stepsPerSecond: { label: "training steps per second" },
  ticksPerSecond: { label: "world ticks per second" },
  envShare: { label: "share of an update spent waiting on worlds", good: "down" },
  rolloutShare: { label: "share of an update spent collecting", good: "down" },
};

// Not charts: the x axis itself, and the things that are one value per record
// rather than a curve.
const NOT_A_SERIES = new Set([
  "step", "episode", "seed", "seedLow", "seedHigh", "mapIndex", "elapsedSeconds", "update",
  "goalStartX", "goalStartY", "goalTargetX", "goalTargetY",
]);
const MOVEMENT_SERIES = new Set([
  "bestBenchmarkSuccess",
  "bestBenchmarkSeconds",
  "benchmarkSuccess",
  "benchmarkReached",
  "benchmarkEpisodes",
  "benchmarkSeconds",
  "benchmarkSpeed",
  "benchmarkEfficiency",
  "benchmarkDistance",
  "benchmarkDetourSuccess",
  "benchmarkDetourReached",
  "benchmarkDetourEpisodes",
  "benchmarkDetourSeconds",
  "benchmarkWallSeconds",
  "validationSuccess",
  "validationSeconds",
  "championRoutes",
  "championSeconds",
  "championDeltaSeconds",
  "championDeltaSecondsLow",
  "championDeltaSecondsHigh",
  "championDeltaSuccess",
  "championDeltaSuccessLow",
  "championDeltaSuccessHigh",
  "championGained",
  "championLost",
  "stabilityStrikes",
  "rollbackCount",
  "testSuccess",
  "testSeconds",
  "testStartDeltaSeconds",
  "testStartDeltaSuccess",
  "bestGoals",
  "goalsReached",
  "goalsMissed",
  "goalsAssigned",
  "goalAssignedDistance",
  "goalSuccess",
  "goalSeconds",
  "goalSpeed",
  "goalPathEfficiency",
  "goalRadiusPx",
  "goalPatience",
  "goalProgressScale",
  "goalsVsPast",
  "goalsMissedVsPast",
  "goalSecondsVsPast",
  "goalSpeedVsPast",
  "goalEfficiencyVsPast",
  "fromGoal",
  "fromGoalSpeed",
]);
const COMBAT_SERIES = new Set([
  "bestCombat",
  "kills",
  "deaths",
  "damageDealt",
  "damageTaken",
  "selfDamage",
  "suicides",
  "damageRatio",
  "combat",
  "killsVsPast",
  "damageVsPast",
  "deathsVsPast",
  "selfDamageVsPast",
  "suicidesVsPast",
  "probeKills",
  "probeDeaths",
  "probeSuicides",
  "probeDamageDealt",
  "probeSelfDamage",
  "probeSeconds",
  "fromDamageDealt",
  "fromDamageTaken",
  "fromKill",
  "fromDeath",
  "fromSuicide",
  "fromApproach",
  "fromOnTarget",
  "fromAimedShot",
]);

const FIGURES = [
  ["step", "steps", (value) => count(value)],
  ["stepsPerSecond", "steps/s", (value) => count(Math.round(value))],
  ["episodeReward", "reward", (value) => value.toFixed(2)],
  [
    "bestBenchmarkSuccess",
    "best fixed test",
    (value) => `${Math.round(value * 100)}%`,
    () => run?.meta?.task === "movement",
  ],
  [
    "benchmarkSuccess",
    "fixed test",
    (value) => `${Math.round(value * 100)}%`,
    () => run?.meta?.task === "movement",
  ],
  ["benchmarkSeconds", "fixed seconds", (value) => value.toFixed(1), () => run?.meta?.task === "movement"],
  [
    "benchmarkDetourSuccess",
    "fixed detours",
    (value) => `${Math.round(value * 100)}%`,
    () => run?.meta?.task === "movement",
  ],
  ["goalsReached", "training goals", (value) => value.toFixed(2), () => run?.meta?.task === "movement"],
  ["goalSeconds", "seconds/goal", (value) => value.toFixed(1), () => run?.meta?.task === "movement"],
  [
    "goalSpeed",
    "goal speed",
    (value) => `${value.toFixed(0)} px/s`,
    () => run?.meta?.task === "movement",
  ],
  [
    "goalPathEfficiency",
    "direct route",
    (value) => `${Math.round(value * 100)}%`,
    () => run?.meta?.task === "movement",
  ],
  [
    "goalPatience",
    "deadline",
    (value) => `${Math.round(value)} steps`,
    () => run?.meta?.task === "movement",
  ],
  ["bestCombat", "best", (value) => value.toFixed(2), () => run?.meta?.task !== "movement"],
  ["kills", "kills", (value) => value.toFixed(2), () => run?.meta?.task !== "movement"],
  ["deaths", "deaths", (value) => value.toFixed(2), () => run?.meta?.task !== "movement"],
  ["probeKills", "vs still", (value) => value.toFixed(2), () => run?.meta?.task !== "movement"],
  [
    "selfDamage",
    "self damage",
    (value) => value.toFixed(0),
    () => run?.meta?.task !== "movement",
  ],
  ["stuckSteps", "stuck", (value) => value.toFixed(0)],
  ["demoFrames", "your frames", (value) => count(value)],
  ["demoAgreement", "agrees with you", (value) => `${(value * 100).toFixed(0)}%`],
];

const STATUS = {
  running: ["running", "true"],
  done: ["finished", null],
  failed: ["failed", "bad"],
  stopped: ["stopped", "waiting"],
};

const count = (value) => {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  return String(Math.round(value));
};

const duration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const whole = Math.floor(seconds);
  const parts = [Math.floor(whole / 3600), Math.floor((whole % 3600) / 60), whole % 60];
  return parts[0] > 0
    ? `${parts[0]}h ${parts[1]}m`
    : parts[1] > 0
      ? `${parts[1]}m ${parts[2]}s`
      : `${parts[2]}s`;
};

let runs = [];
let run = null;
let records = [];
// What `npm run evaluate --history` wrote for the selected run, if anything.
let history = null;
let selected = new URL(location.href).searchParams.get("run") ?? null;
let stream = null;
const charts = new Map();

function subscribe() {
  void pollObserver();
  clearInterval(subscribe.observerTimer);
  subscribe.observerTimer = setInterval(pollObserver, 3000);
  stream?.close();
  const query = selected ? `?run=${encodeURIComponent(selected)}` : "";
  stream = new EventSource(`events${query}`);
  stream.addEventListener("runs", (event) => {
    runs = JSON.parse(event.data).runs;
    if (!selected) selected = runs[0]?.id ?? null;
    run = runs.find((one) => one.id === selected) ?? run;
    if (history?.id !== selected) void loadHistory(selected);
    render();
  });
  stream.addEventListener("run", (event) => {
    run = JSON.parse(event.data).run ?? run;
    render();
  });
  stream.addEventListener("records", (event) => {
    const message = JSON.parse(event.data);
    if (message.run !== selected) return;
    if (message.reset) records = message.records;
    else records = records.concat(message.records);
    render();
  });
  stream.addEventListener("open", () => setStatus());
  stream.addEventListener("error", () => setStatus("Cannot reach the monitor."));
}

function setStatus(problem) {
  const chip = element("status");
  const alert = element("alert");
  if (problem) {
    chip.textContent = "disconnected";
    chip.dataset.tone = "bad";
    alert.textContent = problem;
    alert.hidden = false;
    return;
  }
  alert.hidden = true;
  const [label, tone] = STATUS[run?.status] ?? ["waiting", "waiting"];
  chip.textContent = label;
  if (tone) chip.dataset.tone = tone;
  else delete chip.dataset.tone;
}

function renderRunList() {
  const select = element("runs");
  const wanted = runs.map((one) => one.id).join("|");
  if (select.dataset.signature !== wanted) {
    select.dataset.signature = wanted;
    select.replaceChildren(
      ...runs.map((one) => {
        const option = document.createElement("option");
        option.value = one.id;
        const when = new Date(one.startedAt).toLocaleString();
        option.textContent = `${one.label ?? one.id} · ${when}`;
        return option;
      }),
    );
  }
  if (selected) select.value = selected;
}

/**
 * The most recent value of one field. Read backwards rather than off the last
 * record, because not every record carries every field: a note carries none, and
 * a trainer may report some things once an episode and others once an hour.
 */
/**
 * Whether any worm in a match is an older copy of the policy.
 *
 * Without one there is nothing to measure improvement against: every figure is
 * an average over worms that are all the same weights. The figure itself is
 * written as zero rather than left out, so what the run recorded about itself
 * is what to ask.
 */
function hasOpponents() {
  return Number(run?.meta?.opponents ?? 0) > 0;
}

/**
 * Updates a run needs before "is it still learning?" can mean anything.
 *
 * The critic is scored on how well it predicts an episode's return, and no
 * episode closes until the run has collected a whole episode's worth of
 * decisions — `episodeTicks / frameskip` of them, `rolloutSteps` at a time.
 * Until that happens `explainedVariance` sits at zero for a reason that has
 * nothing to do with the gradients, and every healthy run reads as dead for
 * its first quarter of an hour.
 */
function warmupUpdates() {
  const meta = run?.meta ?? {};
  const decisions = Number(meta.episodeTicks) / Number(meta.frameskip);
  const steps = Number(meta.rolloutSteps);
  if (!Number.isFinite(decisions) || !Number.isFinite(steps) || steps <= 0) return 0;
  return Math.ceil(decisions / steps);
}

/** Asked too soon to answer. Shown and waited on, not reported as a failure. */
const TOO_EARLY = -1;

function latest(key, numeric = true) {
  for (let index = records.length - 1; index >= 0; index--) {
    const value = records[index][key];
    if (value === undefined || value === null) continue;
    if (numeric && !Number.isFinite(value)) continue;
    return value;
  }
  return undefined;
}

function renderHeading() {
  if (!run) {
    element("heading").textContent = "No runs recorded yet.";
    return;
  }
  const started = new Date(run.startedAt);
  const until = run.endedAt ? new Date(run.endedAt) : new Date(run.updatedAt);
  const parts = [
    started.toLocaleTimeString(),
    duration(latest("elapsedSeconds") ?? (until - started) / 1000),
    `${records.length} records`,
  ];
  const map = latest("map", false);
  if (map) parts.push(map);
  if (run.meta?.policy) parts.push(run.meta.policy);
  element("heading").textContent = parts.join(" · ");
}

// ---------------------------------------------------------------------------
// The headline: six questions, answered in words.
//
// Everything below this section is a chart of one quantity, and a page of forty
// charts answers "how is it going" only for somebody who already knows which
// four of them matter. These do the reading: each takes a number, compares it
// with where it was earlier in the run, and says what that means in a sentence.
//
// `read` is the number, `show` prints it, `track` is the series whose direction
// decides the colour, and `say` writes the line underneath.

/**
 * Where a quantity has moved: its recent third against the third before it.
 *
 * Only over the records that actually carry it. Combat figures are written when
 * episodes finish, which at 900 decisions an episode is one record in forty-odd
 * — so thirds of the whole run would compare two windows of mostly nothing and
 * report "waiting for enough episodes" forever.
 */
function movement(key) {
  if (!key) return null;
  const seen = [];
  for (const record of records) {
    const value = record?.[key];
    if (Number.isFinite(value)) seen.push(value);
  }
  if (seen.length < 6) return null;
  const third = Math.floor(seen.length / 3);
  const mean = (from, to) => {
    let sum = 0;
    for (let index = from; index < to; index++) sum += seen[index];
    return sum / (to - from);
  };
  const before = mean(seen.length - third * 2, seen.length - third);
  const after = mean(seen.length - third, seen.length);
  return { before, after, change: (after - before) / Math.max(Math.abs(before), 1e-9) };
}

const percent = (value) => `${Math.round(value * 100)}%`;

/** What the room watcher is seeing, refreshed on its own clock. */
let observer = { live: false, watching: null };

async function pollObserver() {
  try {
    const response = await fetch("observer");
    observer = await response.json();
  } catch {
    observer = { live: false, watching: null };
  }
  renderHeadlines();
}
const CHANGED = 0.08;

const HEADLINES = [
  {
    when: () => run?.meta?.task === "movement",
    title: "How many identical test routes does it solve?",
    good: "up",
    track: "benchmarkSuccess",
    read: () => latest("benchmarkSuccess"),
    show: percent,
    unit: () => {
      const reached = latest("benchmarkReached");
      const episodes = latest("benchmarkEpisodes");
      const seconds = latest("benchmarkSeconds");
      const detourReached = latest("benchmarkDetourReached");
      const detourEpisodes = latest("benchmarkDetourEpisodes");
      return `${Number.isFinite(reached) ? Math.round(reached) : "—"}/`
        + `${Number.isFinite(episodes) ? Math.round(episodes) : "—"} fixed routes · `
        + `${Number.isFinite(seconds) ? seconds.toFixed(1) : "—"}s including failures · `
        + `${Number.isFinite(detourReached) ? Math.round(detourReached) : "—"}/`
        + `${Number.isFinite(detourEpisodes) ? Math.round(detourEpisodes) : "—"} detours`;
    },
    say: (move) => {
      if (!move) return "Every checkpoint receives the same map, spawn and destination in an isolated world.";
      if (move.change > CHANGED) return "Solving more of the unchanged validation routes.";
      if (move.change < -CHANGED) return "Solving fewer of the unchanged validation routes.";
      return "Success on the unchanged validation routes is flat.";
    },
  },
  {
    // The figure that decides best.pt. The totals above move by a few routes
    // between checks of policies that are no different; this pairs the policy
    // with the champion route by route, and says when the gap is more than that.
    when: () => run?.meta?.task === "movement",
    title: "Is it faster than its champion on the same routes?",
    good: "down",
    track: "championDeltaSeconds",
    read: () => latest("championDeltaSeconds"),
    show: (value) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}s`,
    unit: () => {
      const signed = (value) => (Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}` : "—");
      const whole = (value) => (Number.isFinite(value) ? Math.round(value) : "—");
      return `a route, 95% [${signed(latest("championDeltaSecondsLow"))}, `
        + `${signed(latest("championDeltaSecondsHigh"))}] over ${whole(latest("championRoutes"))} routes · `
        + `${whole(latest("championGained"))} gained, ${whole(latest("championLost"))} lost`;
    },
    state: () => {
      if (latest("championDeltaSecondsHigh") < 0) return "good";
      if (latest("championDeltaSecondsLow") > 0) return "bad";
      return "flat";
    },
    say: (move, value) => {
      if (!Number.isFinite(value)) return "The starting policy is the champion until a check beats it.";
      if (latest("stabilityDecision", false) === "promote") {
        return "Faster than the champion by more than checks of the same policy differ; it took its place.";
      }
      if (latest("championDeltaSecondsHigh") < 0) {
        return "Faster overall, but one suite was slower or fewer routes were solved, so the champion stays.";
      }
      if (latest("championDeltaSecondsLow") > 0) {
        return "Slower than the champion by more than checks of the same policy differ.";
      }
      return "No difference from the champion that these routes can show.";
    },
  },
  {
    when: () => run?.meta?.task === "movement",
    title: "How fast is it inside the random training worlds?",
    good: "down",
    track: "goalSeconds",
    read: () => latest("goalSeconds"),
    show: (value) => `${value.toFixed(1)}s`,
    unit: () => {
      const goals = latest("goalsReached");
      const speed = latest("goalSpeed");
      const distance = latest("goalAssignedDistance");
      return `${Number.isFinite(goals) ? goals.toFixed(2) : "—"} training goals · `
        + `${Number.isFinite(distance) ? distance.toFixed(0) : "—"}px assigned · `
        + `${Number.isFinite(speed) ? speed.toFixed(0) : "—"} direct px/s`;
    },
    say: (move) => {
      const efficiency = latest("goalPathEfficiency");
      const route = Number.isFinite(efficiency)
        ? ` ${percent(efficiency)} of its travelled path points straight at the goal.`
        : "";
      if (!move) return `Waiting for enough finished episodes to compare speed.${route}`;
      if (move.change < -CHANGED) return `Reaching destinations faster than earlier in the run.${route}`;
      if (move.change > CHANGED) return `Taking longer than earlier in the run.${route}`;
      return `Arrival time is flat over the last stretch.${route}`;
    },
  },
  {
    when: () => run?.meta?.task === "movement",
    title: "Is the speed deadline still safe?",
    track: "goalSuccess",
    read: () => latest("goalSuccess"),
    show: percent,
    unit: () => {
      const deadline = latest("goalPatience");
      const frameskip = Number(run?.meta?.frameskip);
      const seconds = Number.isFinite(deadline) && Number.isFinite(frameskip)
        ? deadline * frameskip / 60
        : NaN;
      return `of resolved goals reached · ${Number.isFinite(seconds) ? seconds.toFixed(1) + "s" : "—"} deadline`;
    },
    state: (value) => value >= 0.85 ? "good" : value <= 0.5 ? "bad" : "flat",
    say: (move, value) => {
      if (value >= 0.85) return "Reliable enough for the curriculum to tighten the next deadline window.";
      if (value <= 0.5) return "Too many deadlines are being missed; the curriculum will give time back.";
      return "Holding the current deadline while speed catches up.";
    },
  },
  {
    // First, because it is the only one that answers "is this working".
    //
    // Every other figure on this page is averaged over every worm in the
    // match, and when they are all the same policy that average rises whenever
    // the three of them get more reckless together. This is the worms being
    // trained minus the older copies of themselves they are playing — same map,
    // same match, same weapons. It can only go up by actually being better.
    when: () => run?.meta?.task !== "movement",
    title: "Is it beating its past self?",
    good: "up",
    track: "killsVsPast",
    read: () => (hasOpponents() ? latest("killsVsPast") : undefined),
    show: (value) => (value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2)),
    unit: () => {
      const damage = latest("damageVsPast");
      return Number.isFinite(damage)
        ? `kills a match more than its past self, and ${damage >= 0 ? "+" : ""}${damage.toFixed(0)} damage`
        : "kills a match more than its past self";
    },
    state: (value) => (value > 0.05 ? "good" : value < -0.05 ? "bad" : "flat"),
    say: (move, value) => {
      if (!hasOpponents()) {
        return "Every worm is the same policy, so there is nothing here to be better than — the averages below rise whenever all of them get busier. Start a run with --opponents 0.34 to make this mean something.";
      }
      if (!Number.isFinite(value)) return "Waiting for the first episodes to finish.";
      const standing =
        value > 0.05
          ? "Winning against the older copies it is playing."
          : value < -0.05
            ? "Losing to the older copies it is playing."
            : "Level with the older copies it is playing.";
      if (!move) return `${standing} Not enough episodes yet to say which way it is going.`;
      return move.change > CHANGED
        ? `${standing} Pulling further ahead than earlier in the run.`
        : move.change < -CHANGED
          ? `${standing} The gap has been closing.`
          : `${standing} No change over the last stretch.`;
    },
  },
  {
    // The one figure on the page measured against something that never
    // moves: every --probe-every updates the policy plays a few matches on
    // its own against worms that press nothing. Its past self gets better
    // as it does; these do not, so this is the number that can say whether
    // it has learned to find a worm and kill it.
    when: () => run?.meta?.task !== "movement",
    title: "Can it kill a sitting duck?",
    good: "up",
    track: "probeKills",
    read: () => latest("probeKills"),
    show: (value) => value.toFixed(2),
    unit: () => {
      const own = latest("probeSuicides");
      return Number.isFinite(own)
        ? `kills a match against worms that never move, dying ${own.toFixed(2)} times by its own hand`
        : "kills a match against worms that never move";
    },
    // Good only when it kills them more often than it kills itself: three
    // kills a match against worms that cannot shoot back is not much of a
    // result if it died four times getting them.
    state: (value) => {
      const own = latest("probeSuicides");
      if (value >= 1 && (!Number.isFinite(own) || own <= value)) return "good";
      return value < 0.2 ? "bad" : "flat";
    },
    say: (move, value) => {
      if (!Number.isFinite(value)) {
        return "No probe yet. A run probes itself every --probe-every updates; the first lands after that many.";
      }
      const own = latest("probeSuicides");
      const standing =
        value >= 1 && Number.isFinite(own) && own > value
          ? "It finds them and kills them, and kills itself more often than that."
          : value >= 1
            ? "It finds them and kills them."
            : value < 0.2
              ? "It cannot yet find a worm that does not move, or kills itself first."
              : "The odd kill on a target that never moves.";
      if (!move) return `${standing} Not enough probes yet to say which way it is going.`;
      return move.change > CHANGED
        ? `${standing} Better than earlier in the run.`
        : move.change < -CHANGED
          ? `${standing} Worse than earlier in the run.`
          : `${standing} No change over the last stretch.`;
    },
  },
  {
    title: "Is anyone playing?",
    // The one thing the run's own numbers cannot say. A room sits empty for
    // hours, and "no new frames" looks exactly like "the watcher fell over".
    read: () => (observer.watching || observer.live ? 1 : 0),
    show: () => {
      if (!observer.live) return "not watching";
      const who = observer.recording ?? [];
      return who.length ? who.join(", ") : "empty room";
    },
    unit: () => {
      if (!observer.live) {
        return observer.watching
          ? `${observer.watching} stopped reporting — the watcher is not running`
          : "no watcher is connected to a room";
      }
      const room = observer.room ? `${observer.room}` : "a room";
      return `${observer.watching} is watching ${room} · ${observer.mod ?? ""}`.trim();
    },
    state: () => {
      if (!observer.live) return "bad";
      return (observer.recording ?? []).length ? "good" : "flat";
    },
    say: () => {
      if (!observer.live) {
        return "Nothing is being recorded. Start the watcher with npm run record -- --room-url ...";
      }
      const who = observer.recording ?? [];
      const total = observer.samples ?? 0;
      if (!who.length) {
        return `Connected and waiting. Nobody has a living worm, so nothing is being written. ${total.toLocaleString()} frames so far.`;
      }
      const each = Object.entries(observer.byPlayer ?? {})
        .map(([name, count]) => `${name} ${count.toLocaleString()}`)
        .join(", ");
      return `Recording now. ${total.toLocaleString()} frames this session${each ? ` — ${each}` : ""}.`;
    },
  },
  {
    when: () => run?.meta?.task !== "movement",
    title: "Is it fighting?",
    good: "up",
    track: "kills",
    read: () => latest("kills"),
    show: (value) => value.toFixed(2),
    unit: () => {
      const dealt = latest("damageDealt");
      const whose = hasOpponents() ? ", every worm averaged together" : "";
      return Number.isFinite(dealt)
        ? `kills a match, dealing ${dealt.toFixed(0)} damage${whose}`
        : `kills a match${whose}`;
    },
    say: (move) =>
      !move
        ? "Waiting for enough episodes to tell."
        : move.change > CHANGED
          ? "Killing more than it was earlier in the run."
          : move.change < -CHANGED
            ? "Killing less than it was earlier in the run."
            : "No change over the last stretch of the run.",
  },
  {
    when: () => run?.meta?.task !== "movement",
    title: "Is it blowing itself up?",
    good: "down",
    track: "selfDamage",
    read: () => {
      const self = latest("selfDamage");
      const dealt = latest("damageDealt");
      if (!Number.isFinite(self) || !Number.isFinite(dealt)) return undefined;
      return self + dealt > 0 ? self / (self + dealt) : 0;
    },
    show: percent,
    unit: () => {
      const own = latest("suicides");
      const deaths = latest("deaths");
      // How many of its deaths were nobody else's doing: the number the
      // share of damage above turns into.
      return Number.isFinite(own) && Number.isFinite(deaths) && deaths > 0
        ? `of the damage it causes lands on itself; ${percent(own / deaths)} of its deaths are its own doing`
        : "of the damage it causes lands on itself";
    },
    say: (move, value) => {
      const state =
        value > 0.45
          ? "It hurts itself about as much as it hurts anyone else."
          : value > 0.2
            ? "Still costing itself a lot of health."
            : "Mostly hurting the other worms, not itself.";
      if (!move) return state;
      if (move.change < -CHANGED) return `${state} Improving.`;
      if (move.change > CHANGED) return `${state} Getting worse.`;
      return state;
    },
  },
  {
    title: "Can it get around?",
    good: "down",
    track: "stuckSteps",
    read: () => {
      const stuck = latest("stuckSteps");
      const steps = latest("episodeSteps");
      if (!Number.isFinite(stuck) || !Number.isFinite(steps) || steps <= 0) return undefined;
      return stuck / steps;
    },
    show: percent,
    unit: () => "of the match spent unable to move",
    say: (move, value) => {
      const state =
        value > 0.3
          ? "Wedged or pacing for a third of the match."
          : value > 0.15
            ? "Gets stuck regularly."
            : "Moving freely most of the time.";
      if (!move) return state;
      if (move.change < -CHANGED) return `${state} Improving.`;
      if (move.change > CHANGED) return `${state} Getting worse.`;
      return state;
    },
  },
  {
    title: "Has it made up its mind?",
    good: "down",
    track: "entropyShare",
    read: () => latest("entropyShare"),
    show: percent,
    unit: () => "as undecided as pressing keys at random",
    say: (move, value) =>
      value > 0.8
        ? "Still close to mashing buttons — nearly every key is a coin toss."
        : value > 0.5
          ? "Committing to some keys, still experimenting with others."
          : value > 0.2
            ? "Playing deliberately."
            : "Very decided; it has stopped trying new things.",
  },
  {
    title: "Is it still learning?",
    // Not one quantity but the health of the updates themselves: how far the
    // policy moves each time, how much of the batch the clip holds back, and
    // whether the critic can predict anything at all. A run can look alive on
    // every chart above while these say the gradients stopped meaning anything.
    read: () => {
      const kl = latest("approxKL");
      const clipped = latest("clipFraction");
      const explained = latest("explainedVariance");
      if (![kl, clipped, explained].every(Number.isFinite)) return undefined;
      if ((latest("update") ?? 0) < warmupUpdates()) return TOO_EARLY;
      const moving = kl > 0.0015 && clipped > 0.02;
      const predicting = explained > 0.1;
      return moving && predicting ? 1 : moving || predicting ? 0.5 : 0;
    },
    show: (value) =>
      value === TOO_EARLY ? "too early" : value === 1 ? "yes" : value === 0.5 ? "half" : "stalled",
    unit: () => {
      const kl = latest("approxKL");
      const explained = latest("explainedVariance");
      return `moves ${Number.isFinite(kl) ? kl.toFixed(4) : "—"} per update · `
        + `critic explains ${Number.isFinite(explained) ? percent(explained) : "—"}`;
    },
    state: (value) =>
      value === TOO_EARLY ? "flat" : value === 1 ? "good" : value === 0.5 ? "flat" : "bad",
    say: (move, value) =>
      value === TOO_EARLY
        ? `No episode has finished yet, so the critic has nothing to predict `
          + `against. This answers itself around update ${warmupUpdates()}.`
        : value === 1
          ? "Updates are changing the policy, and the critic can predict the reward."
          : value === 0.5
            ? "Half healthy — one of the two has gone flat. Worth a look."
            : "The updates have stopped moving anything. This will not recover on its own.",
  },
  {
    title: "Is it learning from you?",
    good: "up",
    track: "demoFrames",
    read: () => latest("demoFrames"),
    show: (value) =>
      value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(Math.round(value)),
    unit: () => "frames of your own play in the pile",
    say: () => {
      const frames = latest("demoFrames") ?? 0;
      if (!frames) return "Nothing recorded yet. Play a match and it lands here by itself.";
      const weight = latest("demoWeight");
      const agree = latest("demoAgreement");
      const counts = Number.isFinite(weight)
        ? `Counts for ${percent(Math.min(1, weight / 0.05))} of what it could.`
        : "";
      const matching = Number.isFinite(agree)
        ? ` Presses what you pressed ${percent(agree)} of the time.`
        : "";
      return `${counts}${matching}`.trim();
    },
  },
];

function renderHeadlines() {
  const cards = [];
  for (const headline of HEADLINES) {
    if (headline.when && !headline.when()) continue;
    const value = headline.read();
    if (value === undefined) continue;
    const move = movement(headline.track);
    const card = document.createElement("article");
    card.className = "headline";
    const title = document.createElement("h2");
    title.textContent = headline.title;
    const figure = document.createElement("strong");
    figure.textContent = headline.show(value);
    const unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = headline.unit();
    const verdict = document.createElement("p");
    verdict.textContent = headline.say(move, value);
    card.dataset.state = headline.state
      ? headline.state(value)
      : move && headline.good
        ? (headline.good === "up" ? move.change > CHANGED : move.change < -CHANGED)
          ? "good"
          : (headline.good === "up" ? move.change < -CHANGED : move.change > CHANGED)
            ? "bad"
            : "flat"
        : "flat";
    card.append(title, figure, unit, verdict);
    cards.push(card);
  }
  const board = element("headlines");
  board.replaceChildren(...cards);
  board.hidden = cards.length === 0;
}

function renderFigures() {
  const shown = FIGURES.filter(([, , , when]) => !when || when()).map(
    ([key, label, format]) => [label, format, latest(key)],
  ).filter(
    ([, , value]) => value !== undefined,
  );
  element("figures").replaceChildren(
    ...shown.map(([label, format, value]) => {
      const box = document.createElement("dl");
      box.className = "figure";
      const term = document.createElement("dt");
      term.textContent = label;
      const shownValue = document.createElement("dd");
      shownValue.textContent = format(value);
      box.append(term, shownValue);
      return box;
    }),
  );
}

/** Every numeric field the run has mentioned, in the order it first appeared. */
function seriesNames() {
  const names = [];
  for (const record of records) {
    for (const [key, value] of Object.entries(record)) {
      if (NOT_A_SERIES.has(key) || typeof value !== "number") continue;
      const movementRun = run?.meta?.task === "movement";
      if ((!movementRun && MOVEMENT_SERIES.has(key)) || (movementRun && COMBAT_SERIES.has(key))) {
        continue;
      }
      if (!names.includes(key)) names.push(key);
    }
  }
  // Known fields first, in the order they are declared above, so the chart that
  // answers "is it learning" is at the top.
  const known = Object.keys(SERIES).filter((name) => names.includes(name));
  return [...known, ...names.filter((name) => !known.includes(name))];
}

function chartFor(name) {
  let chart = charts.get(name);
  if (chart) return chart;
  const panel = document.createElement("section");
  panel.className = "panel chart";
  const head = document.createElement("div");
  head.className = "panel-head";
  const title = document.createElement("h2");
  title.textContent = SERIES[name]?.label ?? name;
  const trend = document.createElement("span");
  trend.className = "trend dim";
  head.append(title, trend);
  const canvas = document.createElement("canvas");
  panel.append(head, canvas);
  chart = { panel, canvas, trend };
  charts.set(name, chart);
  return chart;
}

/**
 * The charts, in groups, with only the first one open.
 *
 * Forty panels in one column is a page that answers "how is it going" for
 * somebody who already knows which four of them matter. The headlines above do
 * the reading; these are for when one of them says something surprising and you
 * want to see the shape of it. So: the few that bear on whether this is working
 * are open, and the rest are behind a heading you can click.
 *
 * Anything a run starts reporting that is not named here still appears, in the
 * last group — nothing is hidden, only sorted.
 */
const GROUPS = [
  {
    title: "Is it getting better?",
    open: true,
    of: [
      "goalsVsPast", "goalSecondsVsPast", "goalSpeedVsPast", "goalEfficiencyVsPast",
      "benchmarkSuccess", "benchmarkSeconds", "benchmarkSpeed", "benchmarkEfficiency",
      "bestBenchmarkSuccess", "bestBenchmarkSeconds", "benchmarkReached", "benchmarkEpisodes",
      "benchmarkDistance",
      "benchmarkDetourSuccess", "benchmarkDetourSeconds",
      "benchmarkDetourReached", "benchmarkDetourEpisodes",
      "goalsReached", "goalsMissed", "goalSuccess", "goalSeconds", "goalSpeed",
      "goalPathEfficiency", "bestGoals", "killsVsPast", "damageVsPast", "deathsVsPast",
      "selfDamageVsPast", "probeKills", "probeDeaths", "probeSuicides", "combat",
      "bestCombat", "episodeReward", "bestReward", "meanReward", "kills", "deaths",
    ],
  },
  {
    title: "What it is being paid for",
    of: [
      "shaping", "fromDamageDealt", "fromDamageTaken", "fromKill", "fromDeath", "fromSuicide",
      "fromOnTarget", "fromAimedShot", "fromApproach", "fromExplore",
      "fromRevisit", "fromStuck", "fromGoal", "fromGoalSpeed", "goalProgressScale",
    ],
  },
  {
    title: "How the fights go",
    of: ["damageDealt", "damageTaken", "selfDamage", "suicides", "damageRatio", "stuckSteps", "cellsVisited", "ropeShare"],
  },
  {
    title: "Is it still learning?",
    of: [
      "entropy", "entropyShare", "entropyCoef", "approxKL", "explainedVariance",
      "clipFraction", "policyLoss", "valueLoss", "learningRate", "goalPatience", "goalRadiusPx",
    ],
  },
  {
    title: "Learning from recorded play",
    of: ["demoAgreement", "demoFrames", "demoWeight", "bcLoss"],
  },
  {
    title: "How fast it is going",
    of: ["stepsPerSecond", "ticksPerSecond", "envShare", "rolloutShare", "episodeSteps", "episodes", "probeSeconds", "benchmarkWallSeconds"],
  },
];

const openGroups = new Map();

function groupFor(title, open) {
  let group = openGroups.get(title);
  if (group) return group;
  const box = document.createElement("details");
  box.className = "group";
  box.open = open;
  const summary = document.createElement("summary");
  summary.textContent = title;
  const grid = document.createElement("div");
  grid.className = "charts";
  box.append(summary, grid);
  group = { box, grid, summary };
  openGroups.set(title, group);
  return group;
}

function renderCharts() {
  const names = seriesNames();
  const board = element("charts");
  if (!names.length) {
    board.replaceChildren(empty(run ? "Charts appear as records arrive." : ""));
    return;
  }
  const left = new Set(names);
  const boxes = [];
  for (const { title, of, open } of GROUPS) {
    const mine = of.filter((name) => left.has(name));
    for (const name of mine) left.delete(name);
    if (!mine.length) continue;
    const group = groupFor(title, Boolean(open));
    group.summary.textContent = `${title} · ${mine.length}`;
    group.grid.replaceChildren(...mine.map((name) => chartFor(name).panel));
    boxes.push(group.box);
  }
  if (left.size) {
    const rest = [...left];
    const group = groupFor("Everything else", false);
    group.summary.textContent = `Everything else · ${rest.length}`;
    group.grid.replaceChildren(...rest.map((name) => chartFor(name).panel));
    boxes.push(group.box);
  }
  board.replaceChildren(...boxes);
  for (const name of names) drawChart(name);
}

function empty(text) {
  const note = document.createElement("p");
  note.className = "empty";
  note.textContent = text;
  return note;
}

// A window of the most recent points, so a long run stays readable and the
// drawing cost does not grow with it.
const MAX_POINTS = 600;
const MEAN_WINDOW = 20;

function drawChart(name) {
  const { canvas, trend } = chartFor(name);
  const points = [];
  for (const record of records) {
    const value = record[name];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    points.push([record.step ?? points.length, value]);
  }
  const shown = points.slice(-MAX_POINTS);
  const context = canvas.getContext("2d");
  const ratio = globalThis.devicePixelRatio || 1;
  const width = canvas.clientWidth || 340;
  const height = canvas.clientHeight || 132;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!shown.length) return;

  const values = shown.map(([, value]) => value);
  const means = movingAverage(values, MEAN_WINDOW);
  let low = Math.min(...values, ...means);
  let high = Math.max(...values, ...means);
  if (low === high) {
    low -= 1;
    high += 1;
  }
  const pad = (high - low) * 0.08;
  low -= pad;
  high += pad;
  const left = 46;
  const plot = { x: left, y: 6, width: width - left - 6, height: height - 22 };
  const atX = (index) =>
    plot.x + (shown.length === 1 ? plot.width / 2 : (index / (shown.length - 1)) * plot.width);
  const atY = (value) => plot.y + plot.height - ((value - low) / (high - low)) * plot.height;

  context.strokeStyle = "#2c2520";
  context.lineWidth = 1;
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.stroke();
  // Zero is where "better than nothing" starts, so it is worth a line.
  if (low < 0 && high > 0) {
    context.strokeStyle = "#453b33";
    context.beginPath();
    context.moveTo(plot.x, atY(0));
    context.lineTo(plot.x + plot.width, atY(0));
    context.stroke();
  }

  // The raw points, faint: the noise is information too.
  context.strokeStyle = "rgba(143, 185, 232, 0.35)";
  context.beginPath();
  values.forEach((value, index) => {
    const x = atX(index);
    const y = atY(value);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();

  context.strokeStyle = "#f3e6d4";
  context.lineWidth = 1.5;
  context.beginPath();
  means.forEach((value, index) => {
    const x = atX(index);
    const y = atY(value);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();

  context.fillStyle = "#8d8578";
  context.font = "10px ui-monospace, Menlo, monospace";
  context.textAlign = "right";
  context.fillText(round(high), left - 5, plot.y + 8);
  context.fillText(round(low), left - 5, plot.y + plot.height);
  context.textAlign = "left";
  context.fillText(`step ${count(shown[0][0])}`, plot.x + 2, height - 6);
  context.textAlign = "right";
  context.fillText(`${count(shown.at(-1)[0])}`, plot.x + plot.width, height - 6);

  // The trend is the last mean against the one a window ago: the question the
  // chart is there to answer, in one number.
  const latest = means.at(-1);
  const earlier = means.at(-Math.min(means.length, MEAN_WINDOW + 1)) ?? latest;
  const change = latest - earlier;
  trend.textContent = `${round(latest)} (${change >= 0 ? "+" : ""}${round(change)})`;
  const good = SERIES[name]?.good;
  if (!good || Math.abs(change) < 1e-9) delete trend.dataset.tone;
  else trend.dataset.tone = (change > 0) === (good === "up") ? "true" : "bad";
}

function round(value) {
  const size = Math.abs(value);
  if (size >= 1000) return count(value);
  if (size >= 10) return value.toFixed(1);
  if (size >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function movingAverage(values, span) {
  const out = [];
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    sum += values[index];
    if (index >= span) sum -= values[index - span];
    out.push(sum / Math.min(index + 1, span));
  }
  return out;
}

/**
 * The run's progress measured on the field, when it has been measured.
 *
 * `npm run evaluate --history RUN` seats the run's best against each
 * checkpoint it kept and writes history.json beside them; the monitor serves
 * it and this shows it. Nothing here runs the evaluation — the page reads.
 */
async function loadHistory(id) {
  if (!id) return;
  try {
    const response = await fetch(`runs/${encodeURIComponent(id)}/history`);
    const found = response.ok ? await response.json() : { against: [] };
    history = { id, primary: found.primary ?? null, against: found.against ?? [] };
  } catch {
    history = { id, primary: null, against: [] };
  }
  renderHistory();
}

function renderHistory() {
  const rows = history?.id === selected ? history.against : [];
  element("history-panel").hidden = rows.length === 0;
  const primaryName = rows[0]?.primary ?? history?.primary ?? "kills";
  element("history-score").textContent = `${primaryName}, later - earlier`;
  element("history").replaceChildren(
    ...rows.map((entry) => {
      const primary = entry.metrics?.[entry.primary ?? primaryName] ?? {};
      const [low, high] = primary.interval ?? [NaN, NaN];
      const row = document.createElement("tr");
      const cell = (text, tone) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (tone) td.className = tone;
        row.append(td);
      };
      cell(count(entry.steps));
      const diff = Number(primary.difference);
      cell(
        Number.isFinite(diff) ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}` : "—",
        low > 0 ? "up" : high < 0 ? "down" : undefined,
      );
      cell(Number.isFinite(low) ? `[${low >= 0 ? "+" : ""}${low.toFixed(2)}, ${high >= 0 ? "+" : ""}${high.toFixed(2)}]` : "—");
      cell(entry.pairs ? `${entry.pairs.left_ahead} of ${entry.episodes}` : "—");
      return row;
    }),
  );
}

function renderNotes() {
  const notes = records.filter((record) => typeof record.note === "string");
  element("notes-panel").hidden = notes.length === 0;
  element("notes").replaceChildren(
    ...notes.slice(-30).map((record) => {
      const row = document.createElement("li");
      const when = document.createElement("time");
      when.dateTime = record.at;
      when.textContent = new Date(record.at).toLocaleTimeString();
      const text = document.createElement("span");
      text.textContent = record.note;
      row.append(when, text);
      return row;
    }),
  );
}

function renderMeta() {
  const rows = Object.entries(run?.meta ?? {});
  element("meta").replaceChildren(
    ...rows.map(([key, value]) => {
      const row = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = key;
      const shown = document.createElement("span");
      shown.textContent =
        typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);
      row.append(name, shown);
      return row;
    }),
  );
}

/**
 * Puts the best policy of the selected run on a map and opens it.
 *
 * A page of curves says whether the numbers are improving. It does not say
 * whether the thing has learned to play, and the only way to know that is to
 * watch it.
 */
async function watchSelected() {
  const button = element("watch");
  const wanted = selected;
  button.disabled = true;
  button.textContent = "starting…";
  try {
    const response = await fetch(`runs/${encodeURIComponent(wanted)}/watch`, {
      method: "POST",
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? "the viewer did not start");
    window.open(body.url, "wormy-watch");
    button.textContent = run?.meta?.task === "movement" ? "Watch fixed race" : "Watch";
  } catch (error) {
    const alert = element("alert");
    alert.textContent = `Could not start the viewer: ${error.message}`;
    alert.hidden = false;
    button.textContent = run?.meta?.task === "movement" ? "Watch fixed race" : "Watch";
  } finally {
    renderWatchButton();
  }
}

function renderWatchButton() {
  const button = element("watch");
  const checkpoint = run?.checkpoint;
  button.textContent = run?.meta?.task === "movement" ? "Watch fixed race" : "Watch";
  button.disabled = !checkpoint;
  button.title = checkpoint
    ? run?.meta?.task === "movement"
      ? `Race isolated copies of ${checkpoint === "best.pt" ? "the best policy" : "the latest policy"} on the fixed benchmark routes`
      : `Play ${checkpoint === "best.pt" ? "the best policy" : "the latest policy"} of this run and watch it`
    : "This run has no saved policy to watch";
}

function render() {
  setStatus();
  renderRunList();
  renderWatchButton();
  renderHeading();
  renderHeadlines();
  renderFigures();
  renderCharts();
  renderNotes();
  renderMeta();
}

element("watch").addEventListener("click", () => void watchSelected());

element("runs").addEventListener("change", (event) => {
  selected = event.target.value;
  records = [];
  charts.clear();
  history = null;
  renderHistory();
  void loadHistory(selected);
  subscribe();
});

addEventListener("resize", () => {
  for (const name of charts.keys()) drawChart(name);
});

render();
subscribe();
