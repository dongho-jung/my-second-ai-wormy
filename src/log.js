// Console only, and only for things that change. A 20 Hz sampler that prints
// once per sample is a sampler nobody can read.
const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

export function createLogger({ level = "info", scope = null } = {}) {
  const threshold = LEVELS[level] ?? LEVELS.info;
  const emit = (name, event, fields) => {
    if (LEVELS[name] < threshold) return;
    const detail = Object.entries(fields ?? {})
      .filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => `${key}=${value}`)
      .join(" ");
    const line = `[${scope ? `${scope}:` : ""}${event}]${detail ? ` ${detail}` : ""}`;
    (name === "error" || name === "warn" ? console.error : console.log)(line);
  };
  return {
    debug: (event, fields) => emit("debug", event, fields),
    info: (event, fields) => emit("info", event, fields),
    warn: (event, fields) => emit("warn", event, fields),
    error: (event, fields) => emit("error", event, fields),
    child: (name) =>
      createLogger({ level, scope: scope ? `${scope}.${name}` : name }),
  };
}

export function createNullLogger() {
  const nothing = () => {};
  return {
    debug: nothing,
    info: nothing,
    warn: nothing,
    error: nothing,
    child: () => createNullLogger(),
  };
}
