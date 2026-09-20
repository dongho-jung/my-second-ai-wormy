// Just enough PNG to read a Liero map.
//
// The community map pools publish levels as 8-bit palette PNGs rather than as
// .lev files, and a Liero level is one palette index per pixel — the same thing
// in a different wrapper. So this reads the indices straight out and hands them
// over, rather than decoding to RGB and matching colours back afterwards.
//
// Deliberately narrow: 8-bit indexed, no interlacing. Anything else throws
// rather than guessing, because a map silently read as the wrong thing is a
// policy quietly trained on terrain nobody has.
import { inflateSync } from "node:zlib";

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

/** The chunks, in order, without copying the whole file per chunk. */
function* chunks(bytes) {
  for (let at = 8; at + 8 <= bytes.length; ) {
    const length = bytes.readUInt32BE(at);
    const type = bytes.toString("ascii", at + 4, at + 8);
    const from = at + 8;
    yield { type, data: bytes.subarray(from, from + length) };
    at = from + length + 4; // and past the CRC
  }
}

/**
 * Undo the per-scanline filter PNG applies before compressing.
 *
 * Each row is prefixed with the filter it used, and every filter is defined
 * against the pixel to the left and the row above — which is why this cannot be
 * done row-independently or in place.
 */
function unfilter(raw, width, height, bpp) {
  const stride = width * bpp;
  const out = Buffer.alloc(stride * height);
  for (let row = 0; row < height; row++) {
    const filter = raw[row * (stride + 1)];
    const from = row * (stride + 1) + 1;
    const to = row * stride;
    const above = (row - 1) * stride;
    for (let at = 0; at < stride; at++) {
      const value = raw[from + at];
      const left = at >= bpp ? out[to + at - bpp] : 0;
      const up = row > 0 ? out[above + at] : 0;
      const upLeft = row > 0 && at >= bpp ? out[above + at - bpp] : 0;
      let restored;
      switch (filter) {
        case 0: restored = value; break;
        case 1: restored = value + left; break;
        case 2: restored = value + up; break;
        case 3: restored = value + ((left + up) >> 1); break;
        case 4: {
          // Paeth: whichever of the three neighbours the gradient points at.
          const estimate = left + up - upLeft;
          const dLeft = Math.abs(estimate - left);
          const dUp = Math.abs(estimate - up);
          const dUpLeft = Math.abs(estimate - upLeft);
          restored =
            value + (dLeft <= dUp && dLeft <= dUpLeft ? left : dUp <= dUpLeft ? up : upLeft);
          break;
        }
        default:
          throw new Error(`unknown PNG row filter ${filter} on row ${row}`);
      }
      out[to + at] = restored & 0xff;
    }
  }
  return out;
}

/** An 8-bit palette PNG as its palette indices, plus the palette itself. */
export function decodeIndexedPng(bytes) {
  const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (!SIGNATURE.every((byte, at) => buffer[at] === byte)) {
    throw new Error("not a PNG");
  }
  let header = null;
  let palette = null;
  const parts = [];
  for (const { type, data } of chunks(buffer)) {
    if (type === "IHDR") {
      header = {
        width: data.readUInt32BE(0),
        height: data.readUInt32BE(4),
        depth: data[8],
        colour: data[9],
        interlace: data[12],
      };
    } else if (type === "PLTE") {
      palette = data;
    } else if (type === "IDAT") {
      parts.push(data);
    } else if (type === "IEND") {
      break;
    }
  }
  if (!header) throw new Error("PNG has no header");
  if (header.colour !== 3) {
    throw new Error(`PNG colour type ${header.colour}, and only indexed (3) reads as a level`);
  }
  if (header.depth !== 8) throw new Error(`PNG bit depth ${header.depth}, expected 8`);
  if (header.interlace !== 0) throw new Error("interlaced PNG");
  if (!palette) throw new Error("indexed PNG with no palette");
  const raw = inflateSync(Buffer.concat(parts));
  const indices = unfilter(raw, header.width, header.height, 1);
  return { width: header.width, height: header.height, indices, palette };
}
