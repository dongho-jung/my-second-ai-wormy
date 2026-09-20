const element = (id) => document.getElementById(id);

const STATE_LABEL = {
  loading: "loading",
  awaiting_room: "waiting for a room",
  connected: "reading",
  disconnected: "disconnected",
  captcha_required: "CAPTCHA",
  unsupported_client: "unsupported client",
  stale: "state lagging",
  error: "error",
};
const STATE_TONE = {
  connected: "true",
  awaiting_room: "waiting",
  loading: "waiting",
  captcha_required: "waiting",
  stale: "bad",
  error: "bad",
  unsupported_client: "bad",
  disconnected: "bad",
};

// Terrain is drawn from the game's own material table, never from palette
// index ranges: a level picks its own colours, so an index says nothing.
const TERRAIN_COLOURS = {
  ".": "#17110d",
  "#": "#6a4a30",
  X: "#7f8794",
  outside: "#352f2a",
};

// Things worth seeing that are not terrain. A picture of the ground alone says
// nothing about the crate two ledges over or the shell already in the air.
const ENTITY_COLOURS = {
  self: "#f3e6d4",
  enemy: "#ff5a45",
  health: "#78bc7b",
  weapon: "#e8a723",
  pickup: "#d8cdbf",
  shot: "#ffd9a0",
  ownShot: "#8fb9e8",
  rope: "#9a8a76",
  flag: "#bf99f3",
};

// 1 left, 2 right, 4 up, 8 down, 16 fire, 32 jump, 64 shorten, 128 lengthen,
// 256 dig — the engine's own input bitmask for a worm.
const KEY_BITS = [
  ["left", 1],
  ["right", 2],
  ["up", 4],
  ["down", 8],
  ["fire", 16],
  ["jump", 32],
  ["rope-", 64],
  ["rope+", 128],
  ["dig", 256],
];

for (const [label, colour] of [
  ["Air", TERRAIN_COLOURS["."]],
  ["Dirt", TERRAIN_COLOURS["#"]],
  ["Rock", TERRAIN_COLOURS.X],
  ["Outside", TERRAIN_COLOURS.outside],
  ["Self", ENTITY_COLOURS.self],
  ["Opponent", ENTITY_COLOURS.enemy],
  ["Health", ENTITY_COLOURS.health],
  ["Weapon", ENTITY_COLOURS.weapon],
  ["Shot", ENTITY_COLOURS.shot],
  ["Own shot", ENTITY_COLOURS.ownShot],
]) {
  const item = document.createElement("span");
  const swatch = document.createElement("i");
  swatch.style.background = colour;
  item.append(swatch, document.createTextNode(label));
  element("legend").append(item);
}

let state = null;
let patch = null;
let level = null;
// Which level the game says it is playing, so a picture of the previous one is
// never drawn under this one's worms.
let wanted = null;
const levelIdentity = (map) =>
  map ? `${map.name}:${map.width}x${map.height}` : null;

function text(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function fact(label, value) {
  const item = document.createElement("li");
  item.append(document.createTextNode(label));
  const span = document.createElement("span");
  span.textContent = value;
  item.append(span);
  return item;
}

// Leaving the last frame up would keep drawing a level, or a worm, that is no
// longer there.
function blank(canvas) {
  const paint = canvas.getContext("2d");
  paint.fillStyle = TERRAIN_COLOURS["."];
  paint.fillRect(0, 0, canvas.width, canvas.height);
}

const round = (value, digits = 0) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";

/* --- what to draw on top of the ground ---------------------------------- */

function worldEntities(world) {
  const entities = [];
  const lines = [];
  if (!world?.game) return { entities, lines };
  const selfId = world.game.localPlayerId;
  for (const pickup of world.game.pickups ?? [])
    entities.push({
      x: pickup.position.x,
      y: pickup.position.y,
      kind:
        pickup.kind === "health"
          ? "health"
          : pickup.kind === "weapon"
            ? "weapon"
            : "pickup",
      size: 4,
    });
  for (const shot of world.game.projectiles ?? [])
    entities.push({
      x: shot.position.x,
      y: shot.position.y,
      kind: shot.ownerPlayerId === selfId ? "ownShot" : "shot",
      size: shot.kind === "particle" ? 1 : 2,
    });
  if (world.game.flag)
    entities.push({ ...world.game.flag.position, kind: "flag", size: 4 });
  for (const player of world.game.players ?? []) {
    if (!player.worm) continue;
    entities.push({
      x: player.worm.position.x,
      y: player.worm.position.y,
      kind: player.id === selfId ? "self" : "enemy",
      size: 5,
    });
    // A rope is the difference between a worm that can walk away and one being
    // winched somewhere, and it is invisible in every other reading here.
    if (player.worm.rope)
      lines.push({
        x1: player.worm.position.x,
        y1: player.worm.position.y,
        x2: player.worm.rope.position.x,
        y2: player.worm.rope.position.y,
        kind: "rope",
        dashed: !player.worm.rope.attached,
      });
  }
  return { entities, lines };
}

// Both canvases use one canvas pixel per map pixel, so the same world
// coordinates place worms, ropes and shells on either of them: the whole map
// from its corner, the near window from wherever its patch begins.
function overlay(paint, origin, { entities, lines }) {
  paint.lineWidth = 1;
  for (const line of lines) {
    paint.strokeStyle = ENTITY_COLOURS[line.kind];
    paint.setLineDash(line.dashed ? [3, 3] : []);
    paint.beginPath();
    paint.moveTo(line.x1 - origin[0], line.y1 - origin[1]);
    paint.lineTo(line.x2 - origin[0], line.y2 - origin[1]);
    paint.stroke();
  }
  paint.setLineDash([]);
  for (const thing of entities) {
    paint.fillStyle = ENTITY_COLOURS[thing.kind];
    const size = thing.size ?? 3;
    paint.fillRect(
      Math.round(thing.x - origin[0] - size / 2),
      Math.round(thing.y - origin[1] - size / 2),
      size,
      size,
    );
  }
}

/* --- terrain, one canvas pixel per map pixel ----------------------------- */

// Both pictures are palette indices into the same two tables, so one painter
// serves them: the whole level, and the window the game draws around the worm.
// Each is rebuilt only when its read changes — repainting hundreds of thousands
// of pixels on every animation frame is what makes a dashboard stutter.
const surfaces = {
  map: { canvas: document.createElement("canvas"), key: null },
  near: { canvas: document.createElement("canvas"), key: null },
};

const rgb = (hex) => [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16));

// Material flags, not palette indices: a level picks its own colours, so an
// index says nothing. Bit 3 is the background a worm may stand in, bits 0-1 the
// dirt a shot digs through and bit 2 the rock it cannot.
function colourTable(material) {
  const table = new Uint8Array(256 * 3);
  for (let index = 0; index < 256; index++) {
    if (material) {
      const flags = level.materialFlags[index] ?? 0;
      const solid = (flags & 8) === 0;
      table.set(
        rgb(
          !solid
            ? TERRAIN_COLOURS["."]
            : (flags & 3) !== 0
              ? TERRAIN_COLOURS["#"]
              : TERRAIN_COLOURS.X,
        ),
        index * 3,
      );
    } else {
      table.set(level.paletteRgb.slice(index * 3, index * 3 + 3), index * 3);
    }
  }
  return table;
}

function repaint(surface, key, { data, width, height, origin, bounds, colours }) {
  if (surface.key === key) return true;
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  if (bytes.length !== width * height) return false;
  surface.canvas.width = width;
  surface.canvas.height = height;
  const paint = surface.canvas.getContext("2d");
  const image = paint.createImageData(width, height);
  const outside = rgb(TERRAIN_COLOURS.outside);
  for (let row = 0; row < height; row++) {
    const y = origin[1] + row;
    for (let column = 0; column < width; column++) {
      const x = origin[0] + column;
      // Past the edge of the level is not open air, and drawing it as air is
      // how a worm ends up looking like it could walk off the map.
      const beyond = x < 0 || y < 0 || x >= bounds[0] || y >= bounds[1];
      const target = (row * width + column) * 4;
      const source = bytes[row * width + column] * 3;
      image.data[target] = beyond ? outside[0] : colours[source];
      image.data[target + 1] = beyond ? outside[1] : colours[source + 1];
      image.data[target + 2] = beyond ? outside[2] : colours[source + 2];
      image.data[target + 3] = 255;
    }
  }
  paint.putImageData(image, 0, 0);
  surface.key = key;
  return true;
}

function present(canvas, surface, origin, world) {
  if (canvas.width !== surface.canvas.width) canvas.width = surface.canvas.width;
  if (canvas.height !== surface.canvas.height)
    canvas.height = surface.canvas.height;
  const paint = canvas.getContext("2d");
  paint.imageSmoothingEnabled = false;
  paint.drawImage(surface.canvas, 0, 0);
  overlay(paint, origin, worldEntities(world));
  return paint;
}

function drawMap(world) {
  const canvas = element("map");
  const material = element("material").checked;
  const stale = wanted && levelIdentity(level?.map) !== wanted;
  if (!level?.data || stale) {
    text(element("map-scale"), stale ? "loading the level" : "waiting for a level");
    blank(canvas);
    return;
  }
  const { width, height } = level.map;
  if (
    !repaint(surfaces.map, `${level.at}:${material}`, {
      data: level.data,
      width,
      height,
      origin: [0, 0],
      bounds: [width, height],
      colours: colourTable(material),
    })
  )
    return;
  const paint = present(canvas, surfaces.map, [0, 0], world);
  // Where the near window below actually sits on the map.
  const near = patch?.near;
  if (near) {
    paint.strokeStyle = ENTITY_COLOURS.self;
    paint.setLineDash([5, 4]);
    paint.strokeRect(near.origin[0], near.origin[1], ...near.size);
    paint.setLineDash([]);
  }
  text(
    element("map-scale"),
    `${level.map.name ?? "level"} · ${width}x${height} px · 1:1 · ${age(level.at)}`,
  );
}

function drawNear(world) {
  const canvas = element("near");
  const near = patch?.near;
  const material = element("material").checked;
  if (!near?.data || !level?.paletteRgb) {
    text(
      element("near-scale"),
      patch && !near
        ? "no living worm"
        : near
          ? "loading the level"
          : "waiting for a worm",
    );
    blank(canvas);
    return;
  }
  const [width, height] = near.size;
  if (
    !repaint(surfaces.near, `${patch.at}:${material}`, {
      data: near.data,
      width,
      height,
      origin: near.origin,
      bounds: near.bounds,
      colours: colourTable(material),
    })
  )
    return;
  present(canvas, surfaces.near, near.origin, world);
  text(
    element("near-scale"),
    `${width}x${height} px around the worm · 1:1 · ${age(patch.at)}`,
  );
}

const age = (at) =>
  Number.isFinite(at) ? `${Math.max(0, Math.round((Date.now() - at) / 100) / 10)}s ago` : "—";

/* --- readouts ------------------------------------------------------------ */

function renderSurroundings() {
  const node = element("surroundings");
  if (!patch?.near) return node.replaceChildren();
  const { contacts, walk, overhead } = patch;
  const px = (value) => (value === null ? "clear" : `${value} px`);
  node.replaceChildren(
    fact("walk left", walk.left),
    fact("walk right", walk.right),
    fact("ground", px(overhead.groundPx)),
    fact("ceiling", px(overhead.ceilingPx)),
    fact("wall left", px(overhead.leftPx)),
    fact("wall right", px(overhead.rightPx)),
    fact(
      "contacts",
      `↑${contacts.up} ↓${contacts.down} ←${contacts.left} →${contacts.right}`,
    ),
    fact("stepping", contacts.stepping ? "yes" : "no"),
    fact("tick", patch.tick),
  );
}

function renderSelf(world) {
  const node = element("self");
  const keys = element("keys");
  const self = world?.game?.players?.find((player) => player.local);
  if (!self?.worm) {
    node.replaceChildren(fact("worm", self ? "not alive" : "no local player"));
    keys.replaceChildren();
    return;
  }
  const worm = self.worm;
  const weapon = worm.weapons?.[worm.selectedWeapon];
  node.replaceChildren(
    fact("health", worm.health),
    fact("position", `${round(worm.position.x, 1)}, ${round(worm.position.y, 1)}`),
    fact("velocity", `${round(worm.velocity.x, 2)}, ${round(worm.velocity.y, 2)}`),
    fact("facing", worm.facing),
    fact("aim", `${round((worm.aimRadians * 180) / Math.PI, 1)}°`),
    fact(
      "weapon",
      weapon
        ? `${weapon.name} ${weapon.ammo}/${weapon.capacity}${
            weapon.reloadTicksRemaining
              ? ` · reload ${weapon.reloadTicksRemaining}t`
              : ""
          }`
        : "—",
    ),
    fact(
      "rope",
      worm.rope
        ? `${worm.rope.attached ? "attached" : "flying"} ${round(worm.rope.length, 0)} px`
        : "stowed",
    ),
  );
  keys.replaceChildren(
    ...KEY_BITS.map(([label, bit]) => {
      const node = document.createElement("b");
      node.textContent = label;
      node.dataset.down = String(Boolean(worm.keys & bit));
      return node;
    }),
  );
}

function renderPlayers(world) {
  const body = element("players");
  const players = world?.game?.players ?? [];
  body.replaceChildren(
    ...players.map((player) => {
      const row = document.createElement("tr");
      row.dataset.local = String(player.local);
      row.dataset.alive = String(player.alive);
      const weapon = player.worm?.weapons?.[player.worm.selectedWeapon];
      for (const value of [
        player.name,
        player.team ? (player.team === 1 ? "alpha" : "bravo") : "spectator",
        player.score?.display ?? "—",
        player.score ? `${player.score.kills}/${player.score.deaths}` : "—",
        player.worm ? player.worm.health : "—",
        weapon ? `${weapon.name} ${weapon.ammo}` : "—",
        player.worm
          ? `${round(player.worm.position.x)}, ${round(player.worm.position.y)}`
          : "—",
        player.worm
          ? `${round(player.worm.velocity.x, 1)}, ${round(player.worm.velocity.y, 1)}`
          : "—",
        `${player.pingMs ?? "—"} ms`,
      ]) {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      }
      return row;
    }),
  );
}

function render() {
  const status = state?.status ?? "loading";
  const chip = element("status");
  text(chip, STATE_LABEL[status] ?? status);
  chip.dataset.live = STATE_TONE[status] ?? "waiting";
  const game = state?.game;
  text(
    element("heading"),
    game
      ? `${game.room?.mode ?? "?"} · ${game.map?.name ?? "?"} · tick ${game.tick} · ${game.players.length} players · ${state.source?.sampleHz ?? "?"} Hz`
      : "—",
  );
  const trouble =
    state?.message ??
    (status === "awaiting_room"
      ? "Join or create a room in the game window."
      : null);
  element("alert").hidden = !trouble;
  if (trouble) text(element("alert"), trouble);
  drawMap(state);
  drawNear(state);
  renderSurroundings();
  renderSelf(state);
  renderPlayers(state);
}

/* --- feeds --------------------------------------------------------------- */

// State arrives as it is sampled; terrain is pulled on its own slower cadence,
// because the ground moves far less than the worms standing on it do.
function subscribe() {
  const events = new EventSource("/events");
  events.addEventListener("state", (event) => {
    state = JSON.parse(event.data);
    // A round change swaps the level under the worms. Ask for the new one at
    // once rather than drawing the old ground for another poll or two.
    const identity = levelIdentity(state.game?.map);
    if (identity && identity !== wanted) {
      wanted = identity;
      void pull("/map", (value) => (level = value));
      void pull("/terrain", (value) => (patch = value));
    }
  });
  events.onerror = () => {
    // EventSource reconnects on its own; a closed one means the driver stopped.
    if (events.readyState === EventSource.CLOSED) setTimeout(subscribe, 1000);
  };
}

async function pull(path, into) {
  // There is no terrain to ask for until the game is in a room, and asking
  // anyway fills the console with refusals that look like a fault.
  if (state?.status !== "connected") return;
  try {
    const response = await fetch(path);
    if (response.ok) into(await response.json());
  } catch {
    // The driver is restarting or the room is between rounds; keep the last
    // good picture up rather than blanking the dashboard.
  }
}

subscribe();
setInterval(() => void pull("/terrain", (value) => (patch = value)), 200);
setInterval(() => void pull("/map", (value) => (level = value)), 2000);

const frame = () => {
  render();
  requestAnimationFrame(frame);
};
requestAnimationFrame(frame);
