// The official v20 game bundle, running in plain Node: no browser, no server,
// no rendering. It is the same file `src/adapter-v20.js` is locked to, so the
// physics an agent learns here are the physics it will meet in a real room —
// that equality is the whole reason this environment is worth training in, and
// `CLIENT_SHA256` is what keeps it true.
//
// Nothing here patches game logic. The bundle ends with one call that starts
// the page UI; that call is replaced with a line that hands out the classes the
// closure would otherwise keep to itself. Everything before it, which is all of
// the simulation, runs untouched.
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import vm from "node:vm";
import { CLIENT_SHA256 } from "../adapter-v20.js";
import { decodeIndexedPng } from "./png.js";

// `fetch-assets.sh` in the kit downloads exactly these, from the versioned path
// https://www.webliero.com/v/20/ . The site root serves none of them.
export const ASSETS = {
  bundle: "game-v20.orig.js",
  json5: "json5.min.js",
  wasm: "wasm-flate.wasm",
  resources: "res.dat",
};

// Not in the repository: the bundle is the upstream's to ship, not ours.
export const DEFAULT_ENGINE_DIR = new URL(
  "../../artifacts/headless-sim/",
  import.meta.url,
);

// The minified names this project reads, and what each one is. Everything else
// in the bundle stays where it is.
const CLASSES = {
  World: "Pa", //  update() advances one tick; reset(seed) empties it
  Worm: "V", //    one player's worm, driven by its input bitmask
  Level: "Z", //   the terrain: one palette-index byte per pixel
  Rng: "eb", //    the seeded LCG every random decision goes through
  Reader: "H", //  the binary reader the sprite and zip loaders want
  Sprites: "Jc",
  Zip: "ib",
  Mod: "U", //     dj() parses a mod.json5 through the global JSON5
  Wasm: "A", //    Pc(url) loads the inflate module res.dat needs
};

const BOOT_CALL = "u.js()";

/**
 * What a policy is told about the weapon in its hands, and the one pointed at
 * it. Ten numbers per weapon, measured by `npm run weapons` rather than read
 * out of minified settings fields, and scaled so none of them dwarfs the rest.
 *
 * Without these a worm sees "slot 3, four rounds left" and has to learn what
 * slot 3 is separately on every map — and cannot tell a sniper rifle from a
 * grenade until it has thrown one at its own feet.
 */
export const WEAPON_FEATURES = [
  ["speed", 4],
  ["dropPx", 180], //        how far it falls on the way: flat gun or thrown bomb
  ["rangePx", 450],
  ["shots", 40], //          one bullet or a wall of pellets
  ["damageNear", 50],
  ["damageFar", 50], //      near but not far is a close-quarters weapon
  ["selfNear", 50], //       what it does to the worm holding it
  ["reloadTicks", 300],
  ["fireDelay", 100],
  ["capacity", 10],
];

export const WEAPON_FEATURE_COUNT = WEAPON_FEATURES.length;
/** Measured weapon behaviour, one file per mod: id 0 is a different gun in each. */
const weaponProfilesFor = (mod) =>
  new URL(`../../artifacts/weapons.${mod}.json`, import.meta.url);
const HANDOUT = "__wormyEngineClasses";

// The world settings a room lets its host change, under the names the game's
// own settings screen uses. Defaults are the engine's own.
const RULES = {
  bonusDrops: "qd", //          0 none, 1 health and weapons, 2 health only, 3 weapons only
  bonusSpawnTicks: "Pe", //     ticks between bonus drops
  weaponChangeDelay: "mf", //   ticks before a swapped-to weapon may fire
  damageMultiplier: "Te",
  loadingTimes: "le", //        reload time multiplier
};

let loading = null;

/** Resolve one asset inside an engine directory. */
function assetPath(dir, name) {
  return new URL(name, dir);
}

/**
 * Replace the bundle's UI boot call with a line that hands the simulation
 * classes out of the closure. Kept as a pure string transform so the bytes that
 * run can be checksummed before the transform is applied.
 */
export function patchBundle(source) {
  const at = source.lastIndexOf(BOOT_CALL);
  if (at < 0) {
    throw new Error(`boot call ${BOOT_CALL} not found: this is not v20`);
  }
  const handout = Object.entries(CLASSES)
    .map(([name, minified]) => `try{h.${name}=${minified}}catch(e){}`)
    .join("");
  return `${source.slice(0, at)};var h=globalThis.${HANDOUT}={};${handout};${source.slice(at + BOOT_CALL.length)}`;
}

/**
 * The browser surface the bundle touches on its way to the classes. None of it
 * is game logic: the engine itself never draws, so the stubs only have to be
 * inert enough that evaluating the file does not throw.
 */
function installBrowserStubs() {
  if (globalThis.window === globalThis) return;
  const noop = () => {};
  const element = () =>
    new Proxy(
      {
        style: {},
        dataset: {},
        classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
        children: [],
        childNodes: [],
        firstChild: null,
        innerHTML: "",
        value: "",
        checked: false,
        textContent: "",
      },
      {
        // `has` answers yes to everything so feature tests take the DOM path
        // rather than a fallback that wants something we have not stubbed.
        has: () => true,
        get(target, key) {
          if (key === Symbol.toPrimitive || key === "then" || key === Symbol.iterator)
            return undefined;
          if (key in target) return target[key];
          if (key === "querySelectorAll" || key === "getElementsByTagName")
            return () => [];
          if (key === "querySelector" || key === "getElementById" || key === "closest")
            return () => null;
          if (key === "getAttribute") return () => null;
          if (
            key === "firstElementChild" ||
            key === "lastElementChild" ||
            key === "parentElement" ||
            key === "parentNode"
          )
            return element();
          if (
            key === "appendChild" ||
            key === "insertBefore" ||
            key === "removeChild" ||
            key === "replaceChild"
          )
            return (child) => child;
          if (key === "getContext") return () => new Proxy({}, { get: () => noop });
          return noop;
        },
        set(target, key, value) {
          target[key] = value;
          return true;
        },
      },
    );
  globalThis.window = globalThis;
  globalThis.requestAnimationFrame = noop;
  globalThis.cancelAnimationFrame = noop;
  globalThis.localStorage = { getItem: () => null, setItem: noop, removeItem: noop };
  globalThis.document = {
    createElement: element,
    createElementNS: element,
    createTextNode: element,
    body: element(),
    head: element(),
    documentElement: element(),
    querySelector: () => null,
    querySelectorAll: () => [],
    getElementById: () => null,
    addEventListener: noop,
  };
  globalThis.addEventListener = noop;
  globalThis.WebSocket = class {};
  globalThis.Image = class {};
  globalThis.XMLHttpRequest = class {
    open() {}
    send() {}
    addEventListener() {}
  };
  globalThis.RTCPeerConnection = class {};
  globalThis.AudioContext = class {
    createGain() {
      return { connect: noop, gain: {} };
    }
  };
  globalThis.matchMedia = () => ({ matches: false, addEventListener: noop, addListener: noop });
  globalThis.location = {
    search: "?v=20",
    href: "https://www.webliero.com/?v=20",
    hash: "",
    protocol: "https:",
    host: "www.webliero.com",
    hostname: "www.webliero.com",
    pathname: "/",
    origin: "https://www.webliero.com",
    reload: noop,
  };
  globalThis.history = { pushState: noop, replaceState: noop };
  globalThis.screen = { width: 1280, height: 800 };
  globalThis.innerWidth = 1280;
  globalThis.innerHeight = 800;
  globalThis.devicePixelRatio = 1;
}

/** A seeded LCG, the same one the engine uses, for anything outside the world. */
export function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Level generation draws its terrain shape from a Perlin table the engine
 * shuffles with `Math.random`, so a seed alone does not reproduce a map. Lend
 * it a seeded `Math.random` for the call and give the real one straight back.
 */
function withSeededMathRandom(seed, body) {
  const real = Math.random;
  Math.random = makeRng(seed);
  try {
    return body();
  } finally {
    Math.random = real;
  }
}

/**
 * Who hurt whom.
 *
 * With three worms loose in one world, "the other one lost health" is not a
 * reward signal: two of them can be fighting each other while the third does
 * nothing. The engine knows the answer — every hit goes through
 * `worm.ud(world, amount, attackerPlayerId, weaponId)` — but it tells nobody,
 * so this wraps that one method to write down what it applied. It changes no
 * behaviour: the original runs untouched and the wrapper only reads the health
 * either side of it.
 *
 * Kills need no wrapper; `world.$p` is a hook the engine already calls.
 */
function instrumentDamage(Worm) {
  if (Worm.prototype.ud.wormyInstrumented) return;
  const original = Worm.prototype.ud;
  function ud(world, amount, attacker, weapon) {
    const sink = world[DAMAGE_SINK];
    if (sink === undefined) return original.call(this, world, amount, attacker, weapon);
    const before = this.Xa;
    original.call(this, world, amount, attacker, weapon);
    // Flat triples, not objects: this runs inside the tick loop.
    const applied = before - this.Xa;
    if (applied > 0) sink.push(this.H, attacker, applied);
  }
  ud.wormyInstrumented = true;
  Worm.prototype.ud = ud;
}

const DAMAGE_SINK = "__wormyDamage";

/**
 * Start recording hits and kills in one world. `damage` is (victim, attacker,
 * health) triples and `kills` is (victim, killer) pairs, both flat and both
 * cleared by the caller once a step has read them.
 *
 * A worm can be its own attacker: falling damage and its own explosions arrive
 * with its own id, which is exactly the accounting we want.
 */
export function watchDamage(world) {
  const damage = [];
  const kills = [];
  world[DAMAGE_SINK] = damage;
  world.$p = (victim, killer) => kills.push(victim, killer);
  return {
    damage,
    kills,
    clear() {
      damage.length = 0;
      kills.length = 0;
    },
  };
}

/**
 * A dead worm is dropped from `world.za` by the tick that kills it, and
 * `worm.nx()` places a worm without putting it back, so a respawn that only
 * calls `nx` leaves a worm that is never simulated again.
 */
export function respawnWorm(world, worm, loadout) {
  worm.u = true;
  worm.nx(world, loadout);
  if (!world.za.includes(worm)) world.za.push(worm);
  return worm;
}

/**
 * Read the bundle and hand back the simulation. One engine per process: the
 * bundle installs itself on the globals, so a second directory would quietly
 * be the first one's classes.
 */
/**
 * The mod to train and play under: the one the watched room is set to.
 *
 * Not the stock one, and not one res.dat carries. CS Rewormed has a hundred and
 * twenty-nine weapons where Liero 1.33 has forty, under its own constants, and
 * a policy trained on the stock game arrives in that room having learned a
 * different game entirely. The point of running the official bundle headless is
 * that both sides are the same simulation; the mod is half of what that means.
 *
 * It lives on disk rather than in res.dat, so it has to be fetched first —
 * `npm run mods` — the same as the level pool.
 */
export const DEFAULT_MOD = "cs_rewormed";

export function loadEngine({ dir = DEFAULT_ENGINE_DIR, mod = DEFAULT_MOD } = {}) {
  if (loading) return loading;
  loading = (async () => {
    const source = readFileSync(assetPath(dir, ASSETS.bundle), "utf8");
    const sha256 = createHash("sha256").update(source).digest("hex");
    if (sha256 !== CLIENT_SHA256) {
      throw new Error(
        `${ASSETS.bundle} is ${sha256}, not the ${CLIENT_SHA256} this project ` +
          "reads. Re-check the field mappings before changing the checksum.",
      );
    }
    installBrowserStubs();
    // The mod parser reaches for JSON5 through the global, the way the page
    // loads it as a vendor script.
    vm.runInThisContext(
      `${readFileSync(assetPath(dir, ASSETS.json5), "utf8")};globalThis.JSON5=JSON5;`,
      { filename: ASSETS.json5 },
    );
    vm.runInThisContext(patchBundle(source), { filename: ASSETS.bundle });
    const classes = globalThis[HANDOUT];
    const missing = Object.keys(CLASSES).filter((name) => !classes?.[name]);
    if (missing.length) {
      throw new Error(`the v20 bundle did not hand out ${missing.join(", ")}`);
    }

    // res.dat is a zip the engine inflates through a wasm module it fetches.
    const wasm = readFileSync(assetPath(dir, ASSETS.wasm));
    const realFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      arrayBuffer: async () =>
        wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength),
    });
    try {
      await classes.Wasm.Pc(`vendor/${ASSETS.wasm}`);
    } finally {
      globalThis.fetch = realFetch;
    }

    const zip = classes.Zip.read(
      new Uint8Array(readFileSync(assetPath(dir, ASSETS.resources))),
    );
    const settings = loadMod(classes, zip, mod);
    instrumentDamage(classes.Worm);
    return new Engine({
      classes,
      settings,
      sha256,
      dir,
      mod,
      profiles: loadWeaponProfiles(settings, mod),
    });
  })();
  return loading;
}

/** Only for tests that need a second engine: the bundle is process-global. */
export function resetEngineForTesting() {
  loading = null;
}

/**
 * The measured profiles, flattened into one row of scaled numbers per weapon.
 * Missing, every row is zeros and the policy is simply told nothing — better
 * than telling it something invented.
 */
/**
 * A name for each weapon that survives the mod being updated.
 *
 * Ids do not: this mod keeps its crate-only variants at the end of the list, so
 * anything inserted earlier shifts them, and a policy that learned weapon 108
 * would wake up holding something else. Names do not either — twelve names are
 * used twice here, once for the ordinary weapon and once for the strange
 * version that only falls out of a crate. VIRUS 31 spreads a little; VIRUS 108
 * spreads twice as far.
 *
 * So the name is what the weapon is made of: its own definition, hashed. The
 * same weapon after an update keeps its key; a genuinely rebalanced one gets a
 * new key and is learned again, which is the honest outcome.
 */
/**
 * What the room lets each weapon do, read off the room's own weapon screen.
 *
 * The file is the list that screen shows, in order, one `name<TAB>state` per
 * line. Its order is the mod's weapon order, so a line's position is the
 * weapon id — which is the only unambiguous way to say which VIRUS is meant
 * when the mod has two of them and the room enables one and bans the other.
 *
 * "Banned" does not mean absent. In this game it means nobody spawns holding
 * it and it still falls out of crates, which is how the odd ones are kept odd.
 *
 * Checking every name against the loaded mod in order is also the mod-update
 * detector: if a weapon is inserted or renamed upstream, the positions stop
 * lining up and this refuses rather than quietly banning the wrong guns.
 */
export function roomWeapons(settings, mod = DEFAULT_MOD) {
  const path = new URL(`${mod}/room-weapons.txt`, MODS_DIR);
  if (!existsSync(path)) return null;
  const rows = readFileSync(path, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [name, state] = line.split("\t");
      return { name: (name ?? "").trim(), state: (state ?? "").trim() };
    });
  if (rows.length !== settings.O.length) {
    throw new Error(
      `${mod}/room-weapons.txt lists ${rows.length} weapons and the mod has ` +
        `${settings.O.length}. The mod has changed; re-copy the room's weapon list.`,
    );
  }
  const wrong = rows
    .map((row, id) => [id, row.name, String(settings.O[id].name).trim()])
    .filter(([, listed, real]) => listed !== real);
  if (wrong.length) {
    const [id, listed, real] = wrong[0];
    throw new Error(
      `${mod}/room-weapons.txt is out of step with the mod: position ${id} says ` +
        `"${listed}" and the mod has "${real}" (${wrong.length} disagree). ` +
        "Re-copy the room's weapon list.",
    );
  }
  const enabled = [];
  const crateOnly = [];
  rows.forEach((row, id) => (row.state === "Banned" ? crateOnly : enabled).push(id));
  return { enabled, crateOnly };
}

export function weaponKeys(settings) {
  return settings.O.map((weapon) => {
    // Everything but the id, which is the thing that moves.
    const { id, ...rest } = weapon;
    const digest = createHash("sha256")
      .update(JSON.stringify(rest, (key, value) => (key === "id" ? undefined : value)))
      .digest("hex")
      .slice(0, 16);
    return `${String(weapon.name).trim().toUpperCase()}:${digest}`;
  });
}

function loadWeaponProfiles(settings, mod) {
  const features = new Float32Array(settings.O.length * WEAPON_FEATURE_COUNT);
  const path = weaponProfilesFor(mod);
  if (!existsSync(path)) return { features, measured: false, weapons: [] };
  const { weapons } = JSON.parse(readFileSync(path, "utf8"));
  for (const weapon of weapons) {
    if (weapon.id >= settings.O.length) continue;
    WEAPON_FEATURES.forEach(([field, scale], index) => {
      const value = (weapon[field] ?? 0) / scale;
      features[weapon.id * WEAPON_FEATURE_COUNT + index] = Math.max(-2, Math.min(2, value));
    });
  }
  return { features, measured: true, weapons };
}

/** The mod every other one borrows from when it ships no art of its own. */
const BASE_MOD = "liero133";

/** Mods that are not in res.dat, downloaded next to the other artifacts. */
export const MODS_DIR = new URL("../../artifacts/mods/", import.meta.url);

/** Community map pools, which ship PNGs rather than .lev files. */
export const MAPS_DIR = new URL("../../artifacts/maps/", import.meta.url);

/**
 * A mod kept as files rather than inside res.dat.
 *
 * Rooms run community mods — the one this project watches is CS Rewormed, off
 * a GitLab repository — and res.dat only carries the six the client ships. The
 * two halves of the guarantee are the bundle, which is checksummed, and the
 * mod, which is whatever the room is set to; so a mod has to be loadable from
 * wherever it came from, not only from the file the client happens to bundle.
 */
function modOnDisk(mod) {
  const at = new URL(`${mod}/`, MODS_DIR);
  return existsSync(new URL("mod.json5", at)) ? at : null;
}

/** The mods res.dat carries, for telling a typo apart from a missing download. */
const BUNDLED_MODS = ["csliero", "liero133", "nkpromode", "promode", "sorliero", "webliero"];

function loadMod(classes, zip, mod) {
  const onDisk = modOnDisk(mod);
  if (!onDisk && !BUNDLED_MODS.includes(mod)) {
    throw new Error(
      `no mod called ${mod}: it is not in res.dat and not in ` +
        `${new URL(`${mod}/`, MODS_DIR).pathname} — run: npm run mods`,
    );
  }
  if (onDisk) {
    const settings = classes.Mod.dj(readFileSync(new URL("mod.json5", onDisk), "utf8"));
    const art = existsSync(new URL("sprites.wlsprt", onDisk))
      ? readFileSync(new URL("sprites.wlsprt", onDisk))
      : Buffer.from(zip.get(`mods/${BASE_MOD}/sprites.wlsprt`).Ug());
    const sprites = classes.Sprites.read(
      new classes.Reader(
        new DataView(art.buffer, art.byteOffset, art.byteLength),
        true,
      ),
    );
    settings.ba = sprites.ba;
    settings.Ha = sprites.bj;
    return settings.normalize();
  }
  const entry = (name, from = mod) => {
    const found = zip.get(`mods/${from}/${name}`);
    if (!found) throw new Error(`res.dat has no mods/${from}/${name}`);
    return found;
  };
  const settings = classes.Mod.dj(entry("mod.json5").yl());
  // Some mods are rules only — Promode ReRevisited carries a mod.json5 and no
  // sprites at all — and fall back to the stock art, which is what the game
  // itself shows for them. Nothing here draws anything; the sprites are read
  // because the settings want the worm palette that comes with them.
  const art = zip.get(`mods/${mod}/sprites.wlsprt`) ? mod : BASE_MOD;
  const sprites = classes.Sprites.read(
    new classes.Reader(new DataView(entry("sprites.wlsprt", art).Ug()), true),
  );
  settings.ba = sprites.ba;
  settings.Ha = sprites.bj;
  return settings.normalize();
}

export class Engine {
  constructor({ classes, settings, sha256, dir, mod, profiles }) {
    this.classes = classes;
    this.settings = settings;
    this.sha256 = sha256;
    this.dir = dir;
    this.mod = mod;
    this.weaponFeatures = profiles.features;
    this.weaponsMeasured = profiles.measured;
  }

  /** What every weapon id in a loadout means. */
  get weaponNames() {
    return this.settings.O.map((weapon) => weapon.name);
  }

  /** bit 3 background, bits 0-1 diggable dirt, bit 2 rock — one byte per index. */
  get materialFlags() {
    return this.settings.Da;
  }

  /** How many worm colours the mod has before they start repeating. */
  get wormColours() {
    return this.settings.Li?.length || 1;
  }

  /**
   * The generator a room uses when its host asks for a random map: dirt to dig
   * with rock scattered through it. Unlike the stock .lev files, which are
   * nearly all rock, this is terrain an agent can change.
   */
  randomLevel(seed, { width = 504, mirrored = false } = {}) {
    const level = new this.classes.Level();
    withSeededMathRandom(seed, () =>
      level.Jp(new this.classes.Rng(seed), this.settings, width),
    );
    // Mirroring doubles the width, which is how a room makes a map that gives
    // both worms the same ground.
    if (mirrored) level.Mu();
    return level;
  }

  /**
   * A level from a community map: an 8-bit palette PNG.
   *
   * The indices are the terrain. Liero stores one palette index per pixel and
   * so does an indexed PNG, and these pools are drawn against the game's own
   * palette — a spot check finds the same background and rock indices a
   * generated level uses. So the pixels go straight in, with no colour
   * matching to get subtly wrong.
   */
  readPngLevel(name, bytes) {
    const png = decodeIndexedPng(bytes);
    const level = new this.classes.Level();
    level.name = name;
    level.width = png.width;
    level.height = png.height;
    level.data = new Uint8Array(png.indices);
    return level;
  }

  /** Whichever of the two a file is, decided by its name. */
  readAnyLevel(name, bytes) {
    return name.toLowerCase().endsWith(".png")
      ? this.readPngLevel(name, bytes)
      : this.readLevel(name, bytes);
  }

  /** A stock 504x350 Liero level, from the bytes of a .lev file. */
  readLevel(name, bytes) {
    const level = new this.classes.Level();
    const view = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    level.read(name, view);
    return level;
  }

  /**
   * A world of its own: its own copy of the terrain, so parallel worlds dig
   * their own holes, and its own seeded RNG.
   */
  createWorld({ level, seed = 0, rules = {} } = {}) {
    const world = new this.classes.World();
    world.s = this.settings;
    if (level) world.level = level.Ta();
    world.ca.x = seed >>> 0;
    for (const [name, value] of Object.entries(rules)) {
      const field = RULES[name];
      if (!field) {
        throw new Error(
          `unknown world rule ${name}: expected ${Object.keys(RULES).join(", ")}`,
        );
      }
      world[field] = value;
    }
    return world;
  }

  /** The rules a world is actually running, under their settings-screen names. */
  rulesOf(world) {
    return Object.fromEntries(
      Object.entries(RULES).map(([name, field]) => [name, world[field]]),
    );
  }

  /** One worm, owned by `playerId`, carrying five weapons by id. */
  spawnWorm(world, { color = 0, playerId = 0, loadout }) {
    return world.ox(color, playerId, loadout);
  }

  /**
   * Five different weapons drawn from the mod's forty. Variety is the point: a
   * policy that has only ever held a shotgun has learned the shotgun, not the
   * game.
   */
  randomLoadout(random, { slots = 5, pool = null } = {}) {
    const ids = pool ? [...pool] : this.settings.O.map((_, id) => id);
    if (ids.length < slots) {
      throw new Error(`a loadout of ${slots} needs at least that many weapons`);
    }
    for (let index = 0; index < slots; index++) {
      const pick = index + Math.floor(random() * (ids.length - index));
      [ids[index], ids[pick]] = [ids[pick], ids[index]];
    }
    return ids.slice(0, slots);
  }
}
