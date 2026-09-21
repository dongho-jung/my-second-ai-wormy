// One process of worlds, talking to whatever is training them.
//
// The engine is JavaScript and the fastest thing to train a convolution with on
// this machine is PyTorch, so the two have to meet somewhere. They meet here,
// over this process's own stdin and stdout, in fixed binary frames: no port to
// pick, no socket to clean up, and the workers die with whoever started them.
//
// Everything is little-endian and every frame is a four-byte length followed by
// that many bytes. The first frame out is JSON describing the layout of all the
// others; after that it is one observation frame per action frame, forever.
//
// stdout carries frames and nothing else. Anything to say goes to stderr.
import { loadEngine } from "./engine.js";
import { EPISODE_STATS, HEADS, VecWormEnv } from "./vec.js";

const config = JSON.parse(process.argv[2] ?? "{}");

// stdout is the frame channel, and nothing else may touch it. `createLogger`
// writes info lines with console.log, so one log in a helper puts plain text in
// the middle of a binary stream and the reader waits forever on a frame length
// that was really the letters "[bi". Everything chatty goes to stderr instead.
const toStderr = (...parts) =>
  process.stderr.write(`${parts.map(String).join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;


function writeFrame(...parts) {
  let bytes = 0;
  for (const part of parts) bytes += part.byteLength;
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32LE(bytes, 0);
  process.stdout.write(header);
  for (const part of parts) {
    process.stdout.write(
      Buffer.from(part.buffer, part.byteOffset, part.byteLength),
    );
  }
}

/** Calls `onFrame` with each complete frame arriving on stdin. */
function readFrames(stream, onFrame) {
  let held = Buffer.alloc(0);
  stream.on("data", (chunk) => {
    held = held.length ? Buffer.concat([held, chunk]) : chunk;
    for (;;) {
      if (held.length < 4) return;
      const size = held.readUInt32LE(0);
      if (held.length < 4 + size) return;
      const frame = held.subarray(4, 4 + size);
      held = held.subarray(4 + size);
      onFrame(frame);
    }
  });
}

const engine = await loadEngine(config.engine);
const vec = new VecWormEnv(engine, config);
vec.reset();

const layout = {
  ...vec.describe(),
  engineSha256: engine.sha256,
  mod: engine.settings.name,
  statFields: EPISODE_STATS,
  order: ["vectors", "patches", "patches2", "maps", "rewards", "dones", "restarts", "stats"],
  bytes: {
    vectors: vec.vectors.byteLength,
    patches: vec.patches.byteLength,
    // Empty unless a second patch scale was asked for.
    patches2: vec.patches2.byteLength,
    maps: vec.maps.byteLength,
    rewards: vec.rewards.byteLength,
    dones: vec.dones.byteLength,
    // One byte per worm: it came back from the dead this step, so whatever a
    // policy remembered about its last life is about somebody else now.
    restarts: vec.restarts.byteLength,
    stats: vec.stats.byteLength,
  },
  actionBytes: vec.count * vec.agents * HEADS,
};
writeFrame(Buffer.from(JSON.stringify(layout), "utf8"));

const send = () =>
  writeFrame(vec.vectors, vec.patches, vec.patches2, vec.maps, vec.rewards, vec.dones, vec.restarts, vec.stats);

// The first observation is the one the trainer decides on before any action.
send();

readFrames(process.stdin, (frame) => {
  if (frame.length !== layout.actionBytes) {
    process.stderr.write(
      `worker: expected ${layout.actionBytes} action bytes, got ${frame.length}\n`,
    );
    process.exit(2);
  }
  vec.step(new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength));
  send();
});

// A trainer that has gone away takes its workers with it.
process.stdin.on("end", () => process.exit(0));
process.stdin.on("error", () => process.exit(0));
