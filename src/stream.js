import { EventEmitter } from "node:events";
import { setTimeout as sleep } from "node:timers/promises";
import { randomUUID } from "node:crypto";
import { createNullLogger } from "./log.js";

const STUCK_MS = 20_000;

// Samples the observer on a serial loop and republishes each reading as one
// envelope. Serial on purpose: a slow read lowers the real Hz instead of
// stacking CDP calls on top of each other.
export class StateStream extends EventEmitter {
  constructor(
    observer,
    { hz = 20, log = createNullLogger(), now = () => Date.now() } = {},
  ) {
    super();
    this.observer = observer;
    this.hz = hz;
    this.log = log;
    this.now = now;
    this.samples = 0;
    this.startedAt = now();
    this.sessionId = randomUUID();
    this.sequence = 0;
    this.abort = new AbortController();
    this.publish({ status: "loading", game: null });
  }

  publish(value) {
    this.latest = {
      schemaVersion: 1,
      sessionId: this.sessionId,
      sequence: this.sequence++,
      capturedAt: new Date().toISOString(),
      source: {
        game: "webliero",
        clientVersion: 20,
        transport: "cdp",
        sampleHz: this.hz,
      },
      ...value,
    };
    this.observe(this.latest);
    this.emit("state", this.latest);
    return this.latest;
  }

  // Never one line per sample. A status change is worth a record; nothing else
  // is, apart from a run that has been stuck in one status long enough to be a
  // problem the operator should hear about.
  observe(state) {
    this.samples++;
    const now = this.now();
    if (state.status !== this.lastStatus) {
      this.log.info("status", {
        from: this.lastStatus ?? null,
        to: state.status,
        message: state.message ?? null,
        hz: Number(
          (this.samples / Math.max(1, (now - this.startedAt) / 1000)).toFixed(1),
        ),
      });
      this.lastStatus = state.status;
      this.statusSinceAt = now;
      this.stuckWarned = false;
    }
    // A run that sits in loading usually means the game bundle never arrived or
    // never matched. Measured from when this status began, so the brief loading
    // state during shutdown or a reload never trips it.
    const inStatusMs = now - (this.statusSinceAt ?? this.startedAt);
    if (
      !this.stuckWarned &&
      state.status === "loading" &&
      inStatusMs > STUCK_MS
    ) {
      this.stuckWarned = true;
      const problem = this.observer?.problem ?? null;
      this.log.warn("stuck_loading", {
        forMs: inStatusMs,
        hint: problem
          ? "the WebLiero bundle did not match the versioned adapter"
          : this.observer?.verified
            ? "the bundle matched but no game controller was found on the page"
            : "the WebLiero bundle was never seen; check the page loaded",
      });
    }
  }

  start() {
    this.running ??= this.loop();
    return this.running;
  }

  async loop() {
    while (!this.abort.signal.aborted) {
      const start = performance.now();
      const value = await this.observer.read();
      if (this.abort.signal.aborted) break;
      this.publish(value);
      await sleep(
        Math.max(0, 1000 / this.hz - (performance.now() - start)),
        undefined,
        { signal: this.abort.signal },
      ).catch((error) => {
        if (error.name !== "AbortError") throw error;
      });
    }
  }

  async stop() {
    this.abort.abort();
    await this.running?.catch(() => {});
    this.publish({ status: "disconnected", game: null });
  }
}
