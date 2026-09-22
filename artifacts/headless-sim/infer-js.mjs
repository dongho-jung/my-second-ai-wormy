// How long one policy forward pass takes in plain JavaScript — the shape the
// live driver (or the page itself) would run, with no Python in the loop.
const f32 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.1; return a; };

// dense: out = relu(W x + b), W is [out][in] flattened
const dense = (W, b, x, out, relu = true) => {
  const O = out.length, I = x.length;
  for (let o = 0; o < O; o++) {
    let s = b[o], base = o * I;
    for (let i = 0; i < I; i++) s += W[base + i] * x[i];
    out[o] = relu && s < 0 ? 0 : s;
  }
  return out;
};

// conv 3x3 stride s, NCHW, valid padding=1
const conv = (W, b, x, C, H, Wd, K, stride, out) => {
  const OH = Math.floor(H / stride), OW = Math.floor(Wd / stride);
  for (let k = 0; k < K; k++) {
    for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
      let s = b[k];
      for (let c = 0; c < C; c++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const y = oy * stride + dy, xx = ox * stride + dx;
        if (y < 0 || xx < 0 || y >= H || xx >= Wd) continue;
        s += W[((k * C + c) * 3 + (dy + 1)) * 3 + (dx + 1)] * x[(c * H + y) * Wd + xx];
      }
      out[(k * OH + oy) * OW + ox] = s < 0 ? 0 : s;
    }
  }
  return out;
};

const bench = (label, fn, iters = 2000) => {
  for (let i = 0; i < 200; i++) fn();                  // warm JIT
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = (performance.now() - t0) / iters;
  console.log(`${label}: ${ms.toFixed(3)} ms/decision  ->  ${Math.round(1000 / ms).toLocaleString()} decisions/s  (60 Hz uses ${(ms * 60 / 10).toFixed(1)}% of one core)`);
};

// A) vector-only policy: 256 inputs -> 256 -> 256 -> heads
{
  const x = f32(256), h1 = new Float32Array(256), h2 = new Float32Array(256), o = new Float32Array(16);
  const W1 = f32(256 * 256), b1 = f32(256), W2 = f32(256 * 256), b2 = f32(256), W3 = f32(16 * 256), b3 = f32(16);
  bench("A  MLP 256-256-256          ", () => { dense(W1, b1, x, h1); dense(W2, b2, h1, h2); dense(W3, b3, h2, o, false); });
}

// B) realistic: terrain patch 3x32x32 -> conv16 -> conv32 -> concat 128 state -> 256 -> 256 -> heads
{
  const patch = f32(3 * 32 * 32), state = f32(128);
  const c1W = f32(16 * 3 * 9), c1b = f32(16), c1 = new Float32Array(16 * 32 * 32);
  const c2W = f32(32 * 16 * 9), c2b = f32(32), c2 = new Float32Array(32 * 16 * 16);
  const flat = new Float32Array(32 * 16 * 16 + 128);
  const W1 = f32(256 * flat.length), b1 = f32(256), h1 = new Float32Array(256);
  const W2 = f32(256 * 256), b2 = f32(256), h2 = new Float32Array(256);
  const W3 = f32(16 * 256), b3 = f32(16), o = new Float32Array(16);
  const params = 16 * 3 * 9 + 32 * 16 * 9 + 256 * flat.length + 256 * 256 + 16 * 256;
  console.log(`B parameters: ${(params / 1e6).toFixed(2)} M  (${(params * 4 / 1048576).toFixed(1)} MB as float32)`);
  bench("B  conv32x32 + MLP 256-256  ", () => {
    conv(c1W, c1b, patch, 3, 32, 32, 16, 1, c1);
    conv(c2W, c2b, c1, 16, 32, 32, 32, 2, c2);
    flat.set(c2, 0); flat.set(state, c2.length);
    dense(W1, b1, flat, h1); dense(W2, b2, h1, h2); dense(W3, b3, h2, o, false);
  }, 500);
}
