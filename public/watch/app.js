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
  for (const shot of state.projectiles) {
    context.fillStyle = SHOT_COLOUR;
    context.fillRect(Math.round(shot.x), Math.round(shot.y), 1, 1);
  }
  for (const worm of state.worms) {
    if (!worm.alive) continue;
    const colour = WORM_COLOURS[worm.id % WORM_COLOURS.length];
    if (worm.rope) {
      context.strokeStyle = ROPE_COLOUR;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(worm.x, worm.y);
      context.lineTo(worm.rope.x, worm.rope.y);
      context.stroke();
    }
    context.fillStyle = colour;
    context.fillRect(Math.round(worm.x) - 1, Math.round(worm.y) - 2, 3, 5);
    // Where it is aiming, which is most of what a worm is about to do.
    context.strokeStyle = colour;
    context.globalAlpha = 0.6;
    context.beginPath();
    context.moveTo(worm.x, worm.y);
    context.lineTo(worm.x + Math.cos(worm.aim) * 12, worm.y + Math.sin(worm.aim) * 12);
    context.stroke();
    context.globalAlpha = 1;
  }
}

function renderWorms() {
  if (!state) return;
  element("worms").replaceChildren(
    ...state.worms.map((worm) => {
      const row = document.createElement("li");
      row.dataset.dead = String(!worm.alive);
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = WORM_COLOURS[worm.id % WORM_COLOURS.length];
      const name = document.createElement("span");
      name.textContent = worm.alive
        ? `${worm.weapon ?? "—"} ×${worm.ammo}`
        : "waiting to respawn";
      const facts = document.createElement("span");
      facts.className = "facts";
      facts.textContent = `${worm.score.kills}k ${worm.score.deaths}d · ${Math.round(worm.health)} hp`;
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, worm.health))}%`;
      bar.append(fill);
      row.append(swatch, name, facts, bar);
      return row;
    }),
  );
  element("episode").textContent =
    `episode ${state.episode} · seed ${state.seed} · ${(state.elapsedTicks / 60).toFixed(1)}s`;
  element("heading").textContent =
    `${state.map.name} ${state.map.width}×${state.map.height} · ${state.worms.length} worms · ${state.speed}× speed`;
}

function subscribe() {
  const stream = new EventSource("events");
  stream.addEventListener("state", (event) => {
    state = JSON.parse(event.data);
    if (state.levelVersion !== levelVersion) void pullLevel();
    const chip = element("status");
    chip.textContent = "playing";
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
