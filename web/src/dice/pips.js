/**
 * pips.js -- dice-face artwork shared by the 3D cube and the 2D editor.
 *
 * Layouts follow the physical Blind Dice Cube: 2 and 3 run diagonally,
 * 6 is two columns of three, 8 is a ring with an empty middle.
 */

// (row, col) on a 3x3 grid
const LAYOUTS = {
  1: [[1, 1]],
  2: [[0, 2], [2, 0]],
  3: [[0, 2], [1, 1], [2, 0]],
  4: [[0, 0], [0, 2], [2, 0], [2, 2]],
  5: [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]],
  6: [[0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]],
  7: [[0, 0], [1, 0], [2, 0], [1, 1], [0, 2], [1, 2], [2, 2]],
  8: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]],
  9: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
};

/** Draw a die face onto a 2D canvas context of size `size`. */
export function drawPips(ctx, value, size, { bg = "#121214", pip = "#f4f4f0", center = false } = {}) {
  const r = size * 0.14;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = bg;
  roundRect(ctx, size * 0.02, size * 0.02, size * 0.96, size * 0.96, center ? size * 0.3 : r);
  ctx.fill();
  ctx.fillStyle = pip;
  const step = size * 0.27;
  const dot = size * (value === 9 || value === 8 ? 0.075 : 0.085);
  for (const [row, col] of LAYOUTS[value] || []) {
    ctx.beginPath();
    ctx.arc(size / 2 + (col - 1) * step, size / 2 + (row - 1) * step, dot, 0, Math.PI * 2);
    ctx.fill();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const urlCache = new Map();

/** Cached data URL for use in <img> tags. */
export function pipDataUrl(value, opts = {}) {
  const key = value + JSON.stringify(opts);
  if (!urlCache.has(key)) {
    const c = document.createElement("canvas");
    c.width = c.height = 96;
    drawPips(c.getContext("2d"), value, 96, opts);
    urlCache.set(key, c.toDataURL());
  }
  return urlCache.get(key);
}
