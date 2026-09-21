# The learning half, packaged to run somewhere that is not this laptop.
#
# Only the learning half. The watching half drives a real browser through the
# Chrome DevTools Protocol and has nothing to do in a cluster, so playwright —
# the one npm dependency this project has — is never installed, and its browser
# download never happens. Nothing under `src/env/`, `train/` or `scripts/train.js`
# imports anything outside Node's own builtins, so there is no `npm install`
# here at all.
#
# Both runtimes live in one image because the two halves of a step are in
# different languages: the game is JavaScript and runs in `node`, the policy is
# PyTorch. They meet over a worker process's stdin and stdout, so they have to
# be in the same container — there is no port to split them across.
FROM node:26-bookworm-slim

# python3-venv, because `scripts/train.js` looks for the interpreter at
# `artifacts/.venv/bin/python` and hands over to it. Building the venv where it
# already expects to find it keeps `npm run train` meaning the same thing here
# as on a laptop. ca-certificates is for fetching the game's own files below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first and on its own layer: it is most of the image and it changes far
# less often than the code does, so a rebuild after an edit does not pull it
# again. The CPU wheel deliberately — this image is for nodes with no GPU, and
# the general-purpose wheel carries a CUDA runtime that would never be used.
RUN python3 -m venv artifacts/.venv \
 && artifacts/.venv/bin/pip install --no-cache-dir --upgrade pip \
 && artifacts/.venv/bin/pip install --no-cache-dir numpy \
 && artifacts/.venv/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch

COPY package.json ./
COPY src ./src
COPY train ./train
COPY scripts ./scripts
# The training page and the match viewer are served from here.
COPY public ./public

# The game, its mod and the maps the room plays, fetched at build time.
#
# None of this is in the repository — `artifacts/` is gitignored, and the game
# bundle belongs to WebLiero rather than to this project. It is downloaded here
# from https://www.webliero.com/v/20/ , and `fetch-engine` checks the bundle's
# SHA-256 against the one `src/adapter-v20.js` was read off. A build against a
# moved version fails rather than training on a field mapping that no longer
# means what it says.
RUN node scripts/fetch-engine.js \
 && node scripts/fetch-mods.js \
 && node scripts/fetch-maps.js

# Runs are written here. Mount a volume on it to keep checkpoints past the pod;
# mounting it does not hide the engine, the mod or the maps, which are in
# sibling directories.
RUN mkdir -p artifacts/runs artifacts/demos \
 && chown -R node:node /app

USER node

# Reachable from outside the pod, rather than from inside it only.
#
# Both pages bind loopback by default, which is right on a laptop and useless
# behind a Service. These open them onto the pod's interface while leaving the
# host check in place: a request whose Host is neither the bound address nor
# WORMY_PUBLIC_ORIGIN is refused, so widening what may connect does not widen
# what may pretend to be this.
#
# WORMY_PUBLIC_ORIGIN and WORMY_BASE_PATH are set per deployment, not here — an
# image does not know what it will be called. Set them to, say,
# https://dashboard.example.com and /ai-worm , and the pages ask for their own
# files by relative path, so the prefix moves the whole thing at once.
ENV WORMY_HOST=0.0.0.0

# 8768 is the training page. 8769 is the match viewer the Watch button starts,
# which serves the game being played, frame by frame.
EXPOSE 8768 8769

# Overridden in the manifest, where the worker and world counts are matched to
# whatever the node actually has.
CMD ["node", "scripts/train.js", "--help"]
