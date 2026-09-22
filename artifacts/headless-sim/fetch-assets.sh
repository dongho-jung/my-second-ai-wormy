#!/bin/sh
# The three assets the engine needs, from the versioned path. Not from the site
# root: https://www.webliero.com/res.dat is a 404.
set -e
base=https://www.webliero.com/v/20
curl -fsS -o res.dat "$base/res.dat"
curl -fsS -o wasm-flate.wasm "$base/vendor/wasm-flate.wasm"
curl -fsS -o json5.min.js "$base/vendor/json5.min.js"
curl -fsS -o game-v20.orig.js "$base/game-min.js"
ls -la res.dat wasm-flate.wasm json5.min.js game-v20.orig.js
