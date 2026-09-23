// Draws the match the viewer is serving.
//
// The terrain is the level's own palette indices, one byte per pixel, painted
// once per read into an offscreen canvas. It is re-read on a timer rather than
// sent with every frame because the worms dig through it: a static backdrop
// would slowly stop matching what they are standing on.
const element = (id) => document.getElementById(id);

// One colour per worm, in the order the engine spawns them.
const WORM_COLOURS = ["#f3e6d4", "#ff5a45", "#8fb9e8", "#78bc7b", "#e8a723", "#bf99f3"];
const SHOT_COLOUR = "#ffd9a0";
const ROPE_COLOUR = "#9a8a76";
// `goalRadiusPx` in src/env/progress.js: inside this the worm has arrived.
const GOAL_RADIUS_PX = 24;
const LEVEL_MS = 1000;

const arena = element("arena");
const context = arena.getContext("2d");
const terrain = document.createElement("canvas");
const terrainContext = terrain.getContext("2d");

let state = null;
let levelVersion = -1;
let painted = false;

function paintTerrain(level) {
  const { width, height } = level;
  terrain.width = width;
  terrain.height = height;
  arena.width = width;
  arena.height = height;
  const bytes = Uint8Array.from(atob(level.data), (character) => character.charCodeAt(0));
  const image = terrainContext.createImageData(width, height);
  const palette = level.paletteRgb;
  for (let at = 0; at < bytes.length; at++) {
    const index = bytes[at] * 3;
    image.data[at * 4] = palette[index];
    image.data[at * 4 + 1] = palette[index + 1];
    image.data[at * 4 + 2] = palette[index + 2];
    image.data[at * 4 + 3] = 255;
  }
  terrainContext.putImageData(image, 0, 0);
  painted = true;
}

async function pullLevel() {
  try {
    const response = await fetch("level");
    if (!response.ok) return;
    const level = await response.json();
    paintTerrain(level);
    levelVersion = level.version;
  } catch {
    // The match is between episodes or the viewer is going away; the last
    // picture stays up rather than the page blanking.
  }
}

function draw() {
  if (painted) context.drawImage(terrain, 0, 0);
  if (!state) return;
  if (state.race) {
    const { start, goal } = state.race;
    context.strokeStyle = "#e8e0d4";
    context.globalAlpha = 0.45;
    context.lineWidth = 1;
    context.setLineDash([3, 4]);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(goal.x, goal.y);
    context.stroke();
    context.setLineDash([]);
    context.globalAlpha = 1;
    context.fillStyle = "#78bc7b";
    context.fillRect(Math.round(start.x) - 2, Math.round(start.y) - 2, 5, 5);
    context.strokeStyle = "#f3e6d4";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(goal.x, goal.y, GOAL_RADIUS_PX, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = "#f3e6d4";
    context.fillRect(Math.round(goal.x) - 2, Math.round(goal.y) - 2, 5, 5);
  }
  for (const shot of state.projectiles) {
    context.fillStyle = SHOT_COLOUR;
    context.fillRect(Math.round(shot.x), Math.round(shot.y), 1, 1);
  }
  for (const worm of state.worms) {
    if (!worm.alive) continue;
    const colour = WORM_COLOURS[worm.id % WORM_COLOURS.length];
    // Where it is headed, in the worm's own colour, drawn under everything else
    // so a crowd of them does not bury the worms. The circle is the arrival
    // radius the environment actually pays on, not a decoration: inside it the
    // goal is reached and the next one is handed out.
    if (worm.goal && !state.race) {
      context.strokeStyle = colour;
      context.globalAlpha = 0.3;
      context.lineWidth = 1;
      context.setLineDash([2, 3]);
      context.beginPath();
      context.moveTo(worm.x, worm.y);
      context.lineTo(worm.goal.x, worm.goal.y);
      context.stroke();
      context.setLineDash([]);
      context.globalAlpha = 0.7;
      context.beginPath();
      context.arc(worm.goal.x, worm.goal.y, GOAL_RADIUS_PX, 0, Math.PI * 2);
      context.stroke();
      context.fillStyle = colour;
      context.fillRect(Math.round(worm.goal.x) - 1, Math.round(worm.goal.y) - 1, 3, 3);
      context.globalAlpha = 1;
    }
    if (worm.rope) {
      context.strokeStyle = ROPE_COLOUR;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(worm.x, worm.y);
      context.lineTo(worm.rope.x, worm.rope.y);
      context.stroke();
    }
    context.fillStyle = colour;
    context.globalAlpha = worm.finishSeconds === null ? 0.9 : 0.65;
    context.fillRect(Math.round(worm.x) - 1, Math.round(worm.y) - 2, 3, 5);
    // Where it is aiming, which is most of what a worm is about to do.
    context.strokeStyle = colour;
    context.globalAlpha = 0.6;
    context.beginPath();
    context.moveTo(worm.x, worm.y);
    context.lineTo(worm.x + Math.cos(worm.aim) * 12, worm.y + Math.sin(worm.aim) * 12);
    context.stroke();
    context.globalAlpha = 1;
    if (state.race) {
      context.fillStyle = colour;
      context.font = "9px ui-monospace, monospace";
      context.fillText(String(worm.id + 1), Math.round(worm.x) + 4, Math.round(worm.y) - 4);
    }
  }
}

function renderWorms() {
  if (!state) return;
  const race = state.race;
  const direct = race
    ? Math.hypot(race.goal.x - race.start.x, race.goal.y - race.start.y)
    : 0;
  const worms = race
    ? [...state.worms].sort((a, b) => {
        if (a.rank && b.rank) return a.rank - b.rank;
        if (a.rank) return -1;
        if (b.rank) return 1;
        return a.id - b.id;
      })
    : state.worms;
  element("worms").replaceChildren(
    ...worms.map((worm) => {
      const row = document.createElement("li");
      row.dataset.dead = String(!worm.alive && !race);
      row.dataset.finished = String(worm.finishSeconds !== null);
      row.dataset.policyMode = worm.policyMode ?? "";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = WORM_COLOURS[worm.id % WORM_COLOURS.length];
      const name = document.createElement("span");
      name.textContent = race
        ? worm.policyMode === "benchmark"
          ? `Ghost ${worm.id + 1} · benchmark`
          : `Ghost ${worm.id + 1} · sampled`
        : worm.alive
          ? `${worm.weapon ?? "—"} ×${worm.ammo}`
          : "waiting to respawn";
      const facts = document.createElement("span");
      facts.className = "facts";
      const remaining = race
        ? Math.hypot(race.goal.x - worm.x, race.goal.y - worm.y)
        : 0;
      facts.textContent = race
        ? worm.finishSeconds !== null
          ? `#${worm.rank} · ${worm.finishSeconds.toFixed(1)}s`
          : `${Math.round(remaining)}px left`
        : `${worm.score.kills}k ${worm.score.deaths}d · ${Math.round(worm.health)} hp`;
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = race
        ? `${Math.max(0, Math.min(100, (1 - remaining / Math.max(1, direct)) * 100))}%`
        : `${Math.max(0, Math.min(100, worm.health))}%`;
      bar.append(fill);
      row.append(swatch, name, facts, bar);
      return row;
    }),
  );
  element("episode").textContent = race
    ? `route ${race.scenario}/${race.scenarios} · seed ${state.seed} · `
      + `${(state.elapsedTicks / 60).toFixed(1)}s${race.detour ? " · detour" : ""}`
    : `episode ${state.episode} · seed ${state.seed} · ${(state.elapsedTicks / 60).toFixed(1)}s`;
  element("heading").textContent = race
    ? `${state.map.name} ${state.map.width}×${state.map.height} · `
      + `benchmark reference + ${Math.max(0, state.worms.length - 1)} sampled · `
      + `same start and goal · ${state.speed}× speed`
    : `${state.map.name} ${state.map.width}×${state.map.height} · ${state.worms.length} worms · ${state.speed}× speed`;
}

function subscribe() {
  const stream = new EventSource("events");
  stream.addEventListener("state", (event) => {
    state = JSON.parse(event.data);
    if (state.levelVersion !== levelVersion) void pullLevel();
    const chip = element("status");
    chip.textContent = state.race?.done ? "results" : state.race ? "racing" : "playing";
    chip.dataset.tone = "true";
    element("alert").hidden = true;
    renderWorms();
  });
  stream.addEventListener("error", () => {
    const chip = element("status");
    chip.textContent = "disconnected";
    chip.dataset.tone = "bad";
    const alert = element("alert");
    alert.textContent = "The match has stopped. Start another from the training page.";
    alert.hidden = false;
  });
}

void pullLevel();
setInterval(() => void pullLevel(), LEVEL_MS);
subscribe();

const frame = () => {
  draw();
  requestAnimationFrame(frame);
};
requestAnimationFrame(frame);
