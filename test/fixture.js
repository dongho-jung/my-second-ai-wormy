// A small, original fixture shaped like the observed client. No bundled game
// code: every field here is one this project already reads by name.
//
// The level is 64x48 with a dirt floor from y=30 down, a rock block on the far
// left (x 0-5) and a full-height rock wall at x=39. Beyond the wall, from
// x=50 on, the floor is drawn in a colour with no material flags at all: the
// worm stands on it and a rope flies through it, like much of the wall on
// five of the community maps.
export const LEVEL = { width: 64, height: 48, floorY: 30, wallX: 39, ghostX: 50 };

// Palette index 0 background, 1 dirt, 2 rock, 3 the flagless colour. Material
// flags: bit 3 marks the background a worm may stand in, bits 0-1 the dirt a
// weapon digs through and bit 2 the rock it cannot.
function level() {
  const data = new Uint8Array(LEVEL.width * LEVEL.height);
  for (let y = 0; y < LEVEL.height; y++)
    for (let x = 0; x < LEVEL.width; x++) {
      const at = y * LEVEL.width + x;
      if (y >= LEVEL.floorY) data[at] = x >= LEVEL.ghostX ? 3 : 1;
      if (x <= 5) data[at] = 2;
      if (x === LEVEL.wallX) data[at] = 2;
    }
  const materialFlags = new Uint8Array(256).fill(8);
  materialFlags[0] = 8;
  materialFlags[1] = 3;
  materialFlags[2] = 4;
  materialFlags[3] = 0;
  const paletteRgb = new Uint8Array(768);
  paletteRgb.set([10, 10, 10], 0);
  paletteRgb.set([120, 80, 40], 3);
  paletteRgb.set([130, 140, 150], 6);
  return { data, materialFlags, paletteRgb };
}

export function fixture() {
  const { data, materialFlags, paletteRgb } = level();
  const weapon = {
    type: { id: 7, name: "Test cannon", ha: 5, Vi: 120 },
    ha: 0,
    gb: 10,
    wd: -1,
  };
  const worm = {
    u: true,
    x: 32,
    y: 26,
    f: -1.25,
    b: 0.5,
    Xa: 67,
    Wa: 1 | 16,
    direction: 1,
    Oa: 0.25,
    ub: 0.02,
    Ka: 0,
    O: [weapon],
    Fa: { Sc: false },
  };
  const entity = {
    u: true,
    x: 22,
    y: 28,
    f: 2,
    b: -3,
    type: { id: 2 },
    La: 7,
    H: 12,
  };
  const world = {
    level: { name: "Fixture", width: LEVEL.width, height: LEVEL.height, data },
    s: { Ha: paletteRgb, Da: materialFlags, O: [{ name: "Test cannon" }] },
    za: [worm],
    le: 0.4,
    Ib: { $: 2, list: [entity, { ...entity, u: false }, { ...entity, H: 99 }] },
    Zb: { $: 0, list: [] },
    Yb: { $: 1, list: [{ u: true, x: 2, y: 3, kind: 1, lf: 9 }] },
    kc: { $: 0, list: [] },
  };
  return {
    w: { m: { isConnected: true } },
    Ub: {
      sb: 12,
      X: {
        F: world,
        B: new Map([
          [12, { V: 12, S: "Local <script>", qa: 1, Tb: true, dc: 42, ra: worm }],
          [13, { V: 13, S: "Spectator", qa: 0, Tb: false, dc: 31, ra: null }],
        ]),
        Nd: 120,
        jd: 10,
        tc: 15,
        Hc: false,
        yf: "Fixture room",
        Aa: {
          Oi: () => 0,
          Pi: (id) => (id === 12 ? { Xf: 3, kb: "3", va: 5, Ca: 2 } : null),
          Sf: (team) => team + 5,
        },
      },
    },
  };
}
