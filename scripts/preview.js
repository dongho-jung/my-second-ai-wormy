// The dashboard, drawn from the test fixture instead of a live game. No
// browser, no room, no CAPTCHA: just the drawing logic against a level whose
// every pixel is known, which is the only way to tell a rendering bug from a
// game that is simply not in a room yet.
import { snapshotV20 } from "../src/adapter-v20.js";
import { createStateServer } from "../src/server.js";
import { fixture, LEVEL } from "../test/fixture.js";
import { StateStream } from "../src/stream.js";

const controller = fixture();
const worm = controller.Ub.X.B.get(12).ra;
const projectile = controller.Ub.X.F.Ib.list[0];
const start = Date.now();

// Walk the worm back and forth along the floor so the picture moves.
const advance = () => {
  const seconds = (Date.now() - start) / 1000;
  worm.x = 22 + 12 * (1 + Math.sin(seconds / 2));
  worm.y = LEVEL.floorY - 4 - 2 * Math.abs(Math.sin(seconds));
  worm.f = Math.cos(seconds / 2) * 6;
  worm.direction = worm.f >= 0 ? 1 : 0;
  worm.Wa = worm.f >= 0 ? 2 : 1;
  projectile.x = 10 + ((seconds * 20) % 40);
  projectile.y = 14;
  controller.Ub.X.Nd = 120 + Math.round(seconds * 60);
};

const observer = {
  read: async (options = {}) => {
    advance();
    return { status: "connected", game: snapshotV20.call(controller, options) };
  },
};
const stream = new StateStream(observer, { hz: 20 });
const terrain = {
  get near() {
    advance();
    const { tick, localPlayerId, patch } = snapshotV20.call(controller, {
      terrainPatch: true,
    });
    return { at: Date.now(), tick, localPlayerId, near: null, ...patch };
  },
  get map() {
    return { at: Date.now(), ...snapshotV20.call(controller, { terrain: true }) };
  },
  refresh(kind) {
    return this[kind];
  },
};

const server = await createStateServer(stream, {
  port: Number(process.argv[2] ?? 8767),
  terrain,
});
void stream.start();
console.log(`Fixture dashboard: ${server.origin}  (Ctrl+C to stop)`);
process.on("SIGINT", async () => {
  await stream.stop();
  await server.close();
  process.exit(0);
});
