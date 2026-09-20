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
  stream?.close();
  const query = selected ? `?run=${encodeURIComponent(selected)}` : "";
  stream = new EventSource(`/events${query}`);
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

function renderCharts() {
  const names = seriesNames();
  const board = element("charts");
  if (!names.length) {
    board.replaceChildren(empty(run ? "Charts appear as records arrive." : ""));
    return;
  }
  board.replaceChildren(...names.map((name) => chartFor(name).panel));
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
    const response = await fetch(`/runs/${encodeURIComponent(wanted)}/watch`, {
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
