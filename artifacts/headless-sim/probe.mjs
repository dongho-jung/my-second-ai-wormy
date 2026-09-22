// Boot the shipped WebLiero bundle in Node with a fake DOM, only far enough to
// reach the engine classes. Nothing here is the game's own logic.
const noop = () => {};
const mkEl = () => {
  const store = { style: {}, dataset: {}, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    children: [], childNodes: [], firstChild: null, innerHTML: "", value: "", checked: false, textContent: "" };
  const el = new Proxy(store, {
    has: () => true,
    get(t, k) {
      if (k === Symbol.toPrimitive || k === "then" || k === Symbol.iterator) return undefined;
      if (k in t) return t[k];
      if (k === "querySelectorAll" || k === "getElementsByTagName") return () => [];
      if (k === "querySelector" || k === "getElementById" || k === "closest") return () => null;
      if (k === "getAttribute") return () => null;
      if (k === "firstElementChild" || k === "lastElementChild" || k === "parentElement" || k === "parentNode") return mkEl();
      if (k === "appendChild" || k === "insertBefore" || k === "removeChild" || k === "replaceChild") return (c) => c;
      if (k === "getContext") return () => new Proxy({}, { get: () => noop });
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return el;
};
globalThis.window = globalThis;
globalThis.requestAnimationFrame = noop; globalThis.cancelAnimationFrame = noop;
globalThis.localStorage = { getItem: () => null, setItem: noop, removeItem: noop };
globalThis.navigator ??= { userAgent: "node", language: "en" };
globalThis.document = { createElement: mkEl, createElementNS: mkEl, createTextNode: mkEl,
  body: mkEl(), head: mkEl(), documentElement: mkEl(),
  querySelector: () => null, querySelectorAll: () => [], getElementById: () => null, addEventListener: noop };
globalThis.addEventListener = noop; globalThis.WebSocket = class {}; globalThis.Image = class {};
globalThis.performance ??= { now: () => Date.now() };
globalThis.location = { search: "?v=20", href: "https://www.webliero.com/?v=20", hash: "", protocol: "https:", host: "www.webliero.com", hostname: "www.webliero.com", pathname: "/", origin: "https://www.webliero.com", reload: () => {} };
globalThis.history = { pushState: noop, replaceState: noop };
globalThis.screen = { width: 1280, height: 800 };
globalThis.matchMedia = () => ({ matches: false, addEventListener: noop, addListener: noop });
globalThis.AudioContext = class { createGain() { return { connect: noop, gain: {} }; } };
globalThis.fetch = () => Promise.reject(new Error("offline"));
globalThis.XMLHttpRequest = class { open() {} send() {} addEventListener() {} };
globalThis.RTCPeerConnection = class {};
globalThis.innerWidth = 1280; globalThis.innerHeight = 800;
globalThis.devicePixelRatio = 1;
try { await import("./engine.js"); } catch (e) { console.log("boot error:", String(e).split("\n")[0]); }
const wl = globalThis.__wl;
console.log("exported:", wl ? Object.entries(wl).map(([k, v]) => `${k}:${typeof v}`).join(" ") : "NONE");
