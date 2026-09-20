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
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { CLIENT_SHA256 } from "../adapter-v20.js";

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
const HANDOUT = "__wormyEngineClasses";

// The world settings a room lets its host change, under the names the game's
// own settings screen uses. Defaults are the engine's own.
const RULES = {
  bonusDrops: "qd", //          0 none, 1 health and weapons, 2 health only
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
export function loadEngine({ dir = DEFAULT_ENGINE_DIR, mod = "liero133" } = {}) {
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
    return new Engine({ classes, settings, sha256, dir, mod });
  })();
  return loading;
}

/** Only for tests that need a second engine: the bundle is process-global. */
export function resetEngineForTesting() {
  loading = null;
}

function loadMod(classes, zip, mod) {
  const entry = (name) => {
    const found = zip.get(`mods/${mod}/${name}`);
    if (!found) throw new Error(`res.dat has no mods/${mod}/${name}`);
    return found;
  };
  const settings = classes.Mod.dj(entry("mod.json5").yl());
  const sprites = classes.Sprites.read(
    new classes.Reader(new DataView(entry("sprites.wlsprt").Ug()), true),
  );
  settings.ba = sprites.ba;
  settings.Ha = sprites.bj;
  return settings.normalize();
}

export class Engine {
  constructor({ classes, settings, sha256, dir, mod }) {
    this.classes = classes;
    this.settings = settings;
    this.sha256 = sha256;
    this.dir = dir;
    this.mod = mod;
  }

  /** What every weapon id in a loadout means. */
  get weaponNames() {
    return this.settings.O.map((weapon) => weapon.name);
  }

  /** bit 3 background, bits 0-1 diggable dirt, bit 2 rock — one byte per index. */
  get materialFlags() {
    return this.settings.Da;
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
}
