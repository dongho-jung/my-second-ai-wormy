// Watch a trained policy play, in a browser.
//
// The same venv the trainer uses: the policy is a PyTorch checkpoint, so the
// thing that reads it has to be Python. It starts the match itself.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const python = fileURLToPath(new URL("../artifacts/.venv/bin/python", import.meta.url));
const watcher = fileURLToPath(new URL("../train/watch.py", import.meta.url));

if (!existsSync(python)) {
  console.error(
    `No training environment at ${python}\n\n` +
      "Make it once (artifacts/ is gitignored, so it never reaches the repository):\n" +
      "  cd artifacts && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy",
  );
  process.exit(1);
}
const child = spawn(python, [watcher, ...process.argv.slice(2)], { stdio: "inherit" });
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
