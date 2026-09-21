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
  bestReward: { label: "best reward kept", good: "up" },
  kills: { label: "kills", good: "up" },
  deaths: { label: "deaths", good: "down" },
  damageDealt: { label: "damage dealt", good: "up" },
  damageTaken: { label: "damage taken", good: "down" },
  selfDamage: { label: "damage to itself", good: "down" },
  damageRatio: { label: "damage dealt per taken", good: "up" },
  stuckSteps: { label: "steps stuck", good: "down" },
  cellsVisited: { label: "ground covered", good: "up" },
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
  fromExplore: { label: "reward from: new ground", good: "up" },
  fromRevisit: { label: "reward from: doubling back", good: "up" },
  fromStuck: { label: "reward from: being stuck", good: "up" },
  fromGoal: { label: "reward from: the goal", good: "up" },
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
const NOT_A_SERIES = new Set(["step", "episode", "seed", "elapsedSeconds", "update"]);

const FIGURES = [
  ["step", "steps", (value) => count(value)],
  ["stepsPerSecond", "steps/s", (value) => count(Math.round(value))],
  ["episodeReward", "reward", (value) => value.toFixed(2)],
  ["bestReward", "best", (value) => value.toFixed(2)],
  ["kills", "kills", (value) => value.toFixed(2)],
  ["deaths", "deaths", (value) => value.toFixed(2)],
  ["selfDamage", "self damage", (value) => value.toFixed(0)],
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
    // First, because it is the only one that answers "is this working".
    //
    // Every other figure on this page is averaged over every worm in the
    // match, and when they are all the same policy that average rises whenever
    // the three of them get more reckless together. This is the worms being
    // trained minus the older copies of themselves they are playing — same map,
    // same match, same weapons. It can only go up by actually being better.
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
    unit: () => "of the damage it causes lands on itself",
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
  const shown = FIGURES.map(([key, label, format]) => [label, format, latest(key)]).filter(
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
    of: ["killsVsPast", "damageVsPast", "episodeReward", "bestReward", "meanReward", "kills", "deaths"],
  },
  {
    title: "What it is being paid for",
    of: [
      "shaping", "fromDamageDealt", "fromDamageTaken", "fromKill", "fromDeath",
      "fromOnTarget", "fromAimedShot", "fromApproach", "fromExplore",
      "fromRevisit", "fromStuck", "fromGoal",
    ],
  },
  {
    title: "How the fights go",
    of: ["damageDealt", "damageTaken", "selfDamage", "damageRatio", "stuckSteps", "cellsVisited"],
  },
  {
    title: "Is it still learning?",
    of: [
      "entropy", "entropyShare", "entropyCoef", "approxKL", "explainedVariance",
      "clipFraction", "policyLoss", "valueLoss", "learningRate",
    ],
  },
  {
    title: "Learning from recorded play",
    of: ["demoAgreement", "demoFrames", "demoWeight", "bcLoss"],
  },
  {
    title: "How fast it is going",
    of: ["stepsPerSecond", "ticksPerSecond", "envShare", "rolloutShare", "episodeSteps", "episodes"],
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
    button.textContent = "Watch";
  } catch (error) {
    const alert = element("alert");
    alert.textContent = `Could not start the viewer: ${error.message}`;
    alert.hidden = false;
    button.textContent = "Watch";
  } finally {
    renderWatchButton();
  }
}

function renderWatchButton() {
  const button = element("watch");
  const checkpoint = run?.checkpoint;
  button.disabled = !checkpoint;
  button.title = checkpoint
    ? `Play ${checkpoint === "best.pt" ? "the best policy" : "the latest policy"} of this run and watch it`
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
  subscribe();
});

addEventListener("resize", () => {
  for (const name of charts.keys()) drawChart(name);
});

render();
subscribe();
