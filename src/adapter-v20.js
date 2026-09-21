// Field mappings verified against the unmodified official v20 browser client.
//
// `snapshotV20` is stringified and evaluated inside the page through CDP with
// the active game controller as `this`, so it has to stay self-contained: no
// imports, no module scope, no globals of our own, nothing that survives the
// call. It only reads; it never patches or replaces game code.
//
// The mappings below are not an official API. If WebLiero ships a new bundle
// the checksum stops matching and collection refuses to guess — never swap the
// checksum alone.
export const CLIENT_SHA256 =
  "b3c7b33c36510d7ede42f93911a26600a9a5b252ee8670afe384d18017f9a06c";

export function snapshotV20({ terrain = false, terrainPatch = false } = {}) {
  if (!this.w?.m?.isConnected) return null;
  const net = this.Ub;
  const room = net?.X;
  const world = room?.F;
  if (!(room?.B instanceof Map) || !world?.level || !Array.isArray(world.za)) {
    throw new Error("WebLiero v20 state shape no longer matches the adapter");
  }
  const number = (value) => {
    if (!Number.isFinite(value))
      throw new Error("Non-finite or missing game field");
    return value;
  };
  const vector = (x, y) => ({ x: number(x), y: number(y) });
  const level = world.level;
  const map = {
    name: level.name,
    width: number(level.width),
    height: number(level.height),
  };
  const tick = number(room.Nd);

  /* --- the whole map, at pixel resolution -------------------------------- */

  if (terrain) {
    if (
      !(level.data instanceof Uint8Array) ||
      level.data.length !== map.width * map.height ||
      level.data.length > 16_777_216
    ) {
      throw new Error("Invalid or oversized terrain buffer");
    }
    const chunks = [];
    for (let offset = 0; offset < level.data.length; offset += 32768) {
      chunks.push(
        String.fromCharCode(...level.data.subarray(offset, offset + 32768)),
      );
    }
    return {
      tick,
      map,
      encoding: "base64-u8",
      order: "row-major",
      // One byte per pixel, each an index into paletteRgb (r,g,b per entry) and
      // into materialFlags, which says what that material does.
      data: btoa(chunks.join("")),
      paletteRgb: Array.from(level.Ha ?? world.s.Ha),
      materialFlags: Array.from(world.s.Da),
    };
  }

  /* --- what the worm can see around itself ------------------------------- */

  if (terrainPatch) {
    const worm = room.B.get(net.sb)?.ra;
    if (!worm?.u) return { tick, localPlayerId: net.sb, patch: null };
    // A worm can only be where the map says background. Listing the things
    // that block instead kept missing some: the blue (bit 5 alone) and the red
    // pillars (no bits at all) both read as open air. Asking the other
    // question — is this somewhere a worm may be — covers every material the
    // map can hold, including ones never seen yet.
    //
    // A shot is stopped only by rock. It chews through dirt and flies straight
    // over the rest, which is why the two predicates cannot be one.
    const BACKGROUND = 8;
    const SHOT_STOPS = 4;
    const DIGGABLE = 3;
    // Outside the map stops both: the boundary is the hardest wall there is.
    const solidAt = (x, y) => {
      if (x < 0 || y < 0 || x >= level.width || y >= level.height)
        return { solid: true, rock: true, stopsShot: true, diggable: false };
      const flags = world.s.Da[level.data[y * level.width + x]];
      const solid = (flags & BACKGROUND) === 0;
      const diggable = (flags & DIGGABLE) !== 0;
      return {
        // Can a worm be here?
        solid,
        // Terrain a worm cannot remove, which is what a route has to go around.
        rock: solid && !diggable,
        // Will a shot be stopped here? Only rock does that.
        stopsShot: (flags & SHOT_STOPS) !== 0,
        diggable,
      };
    };

    // The near picture: the 426x240 window the game draws, one byte per pixel,
    // exactly as the level stores it. Odd sides, so the worm has a centre pixel
    // and not a seam. Pixels past the level edge are written as background and
    // are told apart by `bounds` rather than by a sentinel index that a level
    // could legitimately use.
    const width = 426 | 1;
    const height = 240 | 1;
    const origin = [
      Math.round(worm.x) - (width >> 1),
      Math.round(worm.y) - (height >> 1),
    ];
    const pixels = new Uint8Array(width * height);
    for (let row = 0; row < height; row++) {
      const y = origin[1] + row;
      if (y < 0 || y >= level.height) continue;
      for (let column = 0; column < width; column++) {
        const x = origin[0] + column;
        if (x < 0 || x >= level.width) continue;
        pixels[row * width + column] = level.data[y * level.width + x];
      }
    }
    const chunks = [];
    for (let offset = 0; offset < pixels.length; offset += 32768)
      chunks.push(
        String.fromCharCode(...pixels.subarray(offset, offset + 32768)),
      );

    // The engine's own contact test: movement on an axis is blocked only when
    // two of that direction's probes are solid, and they sit one pixel away.
    //
    // Inlined, not imported: this whole function is stringified and evaluated
    // inside the page, where module scope does not exist.
    const PROBES = {
      up: [[-1, -4], [0, -4], [1, -4]],
      right: [[1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3]],
      down: [[-1, 4], [0, 4], [1, 4]],
      left: [[-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3]],
    };
    const BLOCKING_CONTACTS = 2;
    const contactCount = (offsets) => {
      let solid = 0;
      for (const [dx, dy] of offsets)
        if (solidAt(Math.round(worm.x) + dx, Math.round(worm.y) + dy).solid)
          solid++;
      return solid;
    };
    const counts = {
      up: contactCount(PROBES.up),
      down: contactCount(PROBES.down),
      left: contactCount(PROBES.left),
      right: contactCount(PROBES.right),
    };
    // The engine lifts a worm one pixel per tick when it has headroom, ground
    // under it and something against its side, which is how a worm gets over a
    // small bump without being told to. It needs ground, so it never fires
    // while the worm is hanging on a rope.
    const stepping =
      counts.up < BLOCKING_CONTACTS &&
      counts.down > 0 &&
      (counts.left > 0 || counts.right > 0);

    // Can the worm take a step that way? Measured across the body's own height
    // rather than from a grid, because a difference of four pixels decides it.
    const WORM_HALF_HEIGHT_PX = 3;
    const STEP_AHEAD_PX = 7;
    const STEP_UP_PX = 7;
    const columnBlock = (x, top, bottom) => {
      let blocked = false;
      let rock = false;
      for (let y = top; y <= bottom; y++) {
        const point = solidAt(x, y);
        if (!point.solid) continue;
        blocked = true;
        if (point.rock) rock = true;
      }
      return { blocked, rock };
    };
    const walkProbe = (direction) => {
      const x = Math.round(worm.x) + direction * STEP_AHEAD_PX;
      const top = Math.round(worm.y) - WORM_HALF_HEIGHT_PX;
      const bottom = Math.round(worm.y) + WORM_HALF_HEIGHT_PX;
      const ahead = columnBlock(x, top, bottom);
      if (!ahead.blocked) return "clear";
      if (!columnBlock(x, top - STEP_UP_PX, bottom - STEP_UP_PX).blocked)
        return "step";
      // Dirt is a door that takes a few seconds to open; rock is a wall.
      return ahead.rock ? "rock" : "dirt";
    };

    // Walking a ray until it meets something solid. Distances, not pictures,
    // are what a rope and a gun are actually decided on.
    const ROPE_RANGE_PX = 300;
    const reach = (dx, dy, limit) => {
      for (let step = 1; step <= limit; step++) {
        const x = Math.round(worm.x + dx * step);
        const y = Math.round(worm.y + dy * step);
        if (x < 0 || y < 0 || x >= level.width || y >= level.height) return null;
        if (solidAt(x, y).solid) return step;
      }
      return null;
    };

    return {
      tick,
      localPlayerId: net.sb,
      patch: {
        self: vector(worm.x, worm.y),
        near: {
          origin,
          size: [width, height],
          // Where the level ends, so a consumer can tell open air from the
          // nothing past the map edge without a second read.
          bounds: [level.width, level.height],
          encoding: "base64-u8",
          order: "row-major",
          data: btoa(chunks.join("")),
        },
        // The engine's own contact counts, verbatim.
        contacts: { ...counts, stepping },
        walk: { left: walkProbe(-1), right: walkProbe(1) },
        overhead: {
          ceilingPx: reach(0, -1, ROPE_RANGE_PX),
          groundPx: reach(0, 1, ROPE_RANGE_PX),
          leftPx: reach(-1, 0, ROPE_RANGE_PX),
          rightPx: reach(1, 0, ROPE_RANGE_PX),
          ropeRangePx: ROPE_RANGE_PX,
        },
        legend:
          "what the worm can see: the 426x240 window the game draws around it, one byte per pixel with the worm on the centre one. Each byte indexes the palette and the material table that /map carries: bit 3 of a material is the background a worm may stand in, bits 0-1 the dirt a shot digs through, bit 2 the rock it cannot",
        walkLegend:
          "one step left or right from where the worm stands: clear, step (a ledge it can walk up onto), dirt (blocked, but diggable) or rock (blocked for good)",
        overheadLegend:
          "pixels from self to the first solid pixel each way; null means nothing within ropeRangePx",
      },
    };
  }

  /* --- the state --------------------------------------------------------- */

  const modeId = room.Aa.Oi();
  const players = [...room.B.values()].map((player) => {
    const worm = player.ra?.u ? player.ra : null;
    const score = room.Aa.Pi(player.V);
    return {
      id: player.V,
      name: player.S,
      team: player.qa,
      admin: player.Tb,
      local: player.V === net.sb,
      pingMs: player.dc,
      score: score
        ? {
            value: score.Xf,
            display: score.kb,
            kills: score.va,
            deaths: score.Ca,
          }
        : null,
      alive: worm !== null,
      worm: worm
        ? {
            position: vector(worm.x, worm.y),
            velocity: vector(worm.f, worm.b),
            health: number(worm.Xa),
            // The engine's own input bitmask for this worm: 1 left, 2 right,
            // 4 up, 8 down, 16 fire, 32 jump, 64 shorten, 128 lengthen,
            // 256 dig. Absent on a replicated worm that has not carried one.
            ...(Number.isFinite(worm.Wa) ? { keys: worm.Wa } : {}),
            facing: worm.direction === 1 ? "right" : "left",
            aimRadians:
              worm.direction === 1
                ? -number(worm.Oa)
                : Math.PI + number(worm.Oa),
            // How fast that angle is turning, per tick, in the same screen
            // angle. Aiming accelerates in this engine (`Oa += ub` each tick),
            // so where the aim will be next tick is not something the angle
            // alone can say. A replicated worm may not carry it.
            aimVelocity:
              worm.direction === 1
                ? -(Number.isFinite(worm.ub) ? worm.ub : 0)
                : Number.isFinite(worm.ub) ? worm.ub : 0,
            selectedWeapon: worm.Ka,
            weapons: worm.O.map((weapon, slot) => ({
              slot,
              id: weapon.type.id,
              name: weapon.type.name,
              ammo: number(weapon.ha),
              capacity: number(weapon.type.ha),
              reloadTicksElapsed: number(weapon.gb),
              reloadTicksRemaining:
                weapon.ha <= 0
                  ? Math.max(
                      0,
                      Math.ceil(
                        number(weapon.type.Vi) * number(world.le) - weapon.gb,
                      ),
                    )
                  : 0,
              cooldownTicksRemaining: Math.max(0, number(weapon.wd)),
            })),
            rope: worm.Fa.Sc
              ? {
                  position: vector(worm.Fa.x, worm.Fa.y),
                  attached: worm.Fa.jc,
                  length: number(worm.Fa.length),
                  anchorPlayerId: worm.Fa.anchor?.H ?? null,
                }
              : null,
          }
        : null,
    };
  });
  const pool = (value, serialize) => {
    if (
      !Array.isArray(value?.list) ||
      !Number.isInteger(value.$) ||
      value.$ < 0 ||
      value.$ > value.list.length
    ) {
      throw new Error("Invalid entity pool");
    }
    return value.list
      .slice(0, value.$)
      .flatMap((entity, slot) => (entity.u ? [serialize(entity, slot)] : []));
  };
  const projectile = (kind) => (entity, slot) => ({
    kind,
    slot,
    typeId: entity.type.id,
    weaponId: entity.La === 255 ? null : entity.La,
    ownerPlayerId: entity.H < 0 ? null : entity.H,
    position: vector(entity.x, entity.y),
    velocity: vector(entity.f, entity.b),
  });
  return {
    tick,
    simulationHz: 60,
    localPlayerId: net.sb,
    room: {
      name: room.yf,
      // Which game this room is actually running. A policy trained on one mod
      // and shown recordings from another is being taught a different set of
      // weapons under a different set of constants, and nothing about the
      // observation says so — it is the same 144 numbers either way.
      mod: world.s?.name ?? null,
      modeId,
      mode:
        ["deathmatch", "last_man_standing", "hold_the_flag", "team_deathmatch"][
          modeId
        ] ?? "unknown",
    },
    match: {
      ended: room.Hc,
      elapsedSeconds: tick / 60,
      timeLimitSeconds: room.jd === 0 ? null : room.jd * 60,
      remainingSeconds:
        room.jd === 0 ? null : Math.max(0, room.jd * 60 - tick / 60),
      scoreLimit: room.tc,
      teamScores:
        modeId === 3 ? { alpha: room.Aa.Sf(1), bravo: room.Aa.Sf(2) } : null,
    },
    map,
    players,
    projectiles: [
      ...pool(world.Ib, projectile("weapon")),
      ...pool(world.Zb, projectile("particle")),
    ],
    pickups: pool(world.Yb, (entity, slot) => ({
      slot,
      kind: entity.kind === 1 ? "health" : "weapon",
      weaponId: entity.kind === 1 ? null : entity.lf,
      weaponName:
        entity.kind === 1 ? null : (world.s.O?.[entity.lf]?.name ?? null),
      position: vector(entity.x, entity.y),
    })),
    flag:
      modeId === 2 && world.kc.list[0]?.u
        ? {
            position: vector(world.kc.list[0].x, world.kc.list[0].y),
            carrierPlayerId: room.Aa.Ad < 0 ? null : room.Aa.Ad,
          }
        : null,
  };
}
