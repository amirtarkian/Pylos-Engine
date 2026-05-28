/**
 * camera-bindings.js -- pure helpers for camera control bindings.
 *
 * Bindings shape:
 *   {
 *     version: 1,
 *     mouse: { left: "orbit"|"pan"|"fly"|"none", middle: ..., right: ... },
 *     fly:   { forward, back, left, right, up, down, boost }   // KeyboardEvent.code strings
 *   }
 */

export const STORAGE_KEY = "pylos.camera.bindings";
export const CURRENT_VERSION = 1;
export const MOUSE_ACTIONS = ["orbit", "pan", "fly", "none"];
export const FLY_KEYS = ["forward", "back", "left", "right", "up", "down", "boost"];

export const DEFAULT_BINDINGS = Object.freeze({
  version: CURRENT_VERSION,
  mouse: { left: "orbit", middle: "pan", right: "fly" },
  fly: {
    forward: "KeyW",
    back: "KeyS",
    left: "KeyA",
    right: "KeyD",
    up: "KeyE",
    down: "KeyQ",
    boost: "ShiftLeft",
  },
});

export function validateBindings(b) {
  if (!b || typeof b !== "object") return false;
  if (b.version !== CURRENT_VERSION) return false;
  if (!b.mouse || typeof b.mouse !== "object") return false;
  for (const btn of ["left", "middle", "right"]) {
    if (!MOUSE_ACTIONS.includes(b.mouse[btn])) return false;
  }
  if (!b.fly || typeof b.fly !== "object") return false;
  for (const k of FLY_KEYS) {
    if (typeof b.fly[k] !== "string" || b.fly[k].length === 0) return false;
  }
  return true;
}

export function loadBindings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return cloneDefaults();
    const parsed = JSON.parse(raw);
    if (!validateBindings(parsed)) return cloneDefaults();
    return parsed;
  } catch (_e) {
    return cloneDefaults();
  }
}

export function saveBindings(b) {
  if (!validateBindings(b)) throw new Error("invalid bindings");
  localStorage.setItem(STORAGE_KEY, JSON.stringify(b));
}

function cloneDefaults() {
  return JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
}
