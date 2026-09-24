// Start the trainer with the Python that has torch in it.
//
// The trainer is Python because the fastest thing on this machine for a
// convolution is PyTorch on the GPU; the environment is JavaScript because the
// game is. This finds the one venv the two of them meet in and hands over.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const python = fileURLToPath(new URL("../artifacts/.venv/bin/python", import.meta.url));
// `--algorithm genetic` evolves the weights instead of following a gradient.
const genetic = process.argv.slice(2).some((one, at, all) =>
  one === "--algorithm=genetic" || (one === "--algorithm" && all[at + 1] === "genetic"));
const trainer = fileURLToPath(new URL(genetic ? "../train/evolve.py" : "../train/ppo.py", import.meta.url));

if (!existsSync(python)) {
  console.error(
    `No training environment at ${python}\n\n` +
      "Make it once (artifacts/ is gitignored, so it never reaches the repository):\n" +
      "  cd artifacts && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy",
  );
  process.exit(1);
}
const child = spawn(python, [trainer, ...process.argv.slice(2)], { stdio: "inherit" });
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
