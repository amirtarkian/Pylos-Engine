// Run with: bun test web/tests/camera-bindings.test.js
import { test, expect, beforeEach } from "bun:test";
import {
  DEFAULT_BINDINGS,
  validateBindings,
  loadBindings,
  saveBindings,
  STORAGE_KEY,
} from "../src/camera-bindings.js";

// Minimal localStorage mock for Node/Bun.
class MemStore {
  constructor() { this.m = new Map(); }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
  setItem(k, v) { this.m.set(k, String(v)); }
  removeItem(k) { this.m.delete(k); }
  clear() { this.m.clear(); }
}

beforeEach(() => {
  globalThis.localStorage = new MemStore();
});

test("DEFAULT_BINDINGS is the Unity preset", () => {
  expect(DEFAULT_BINDINGS.mouse).toEqual({ left: "orbit", middle: "pan", right: "fly" });
  expect(DEFAULT_BINDINGS.fly.forward).toBe("KeyW");
  expect(DEFAULT_BINDINGS.fly.back).toBe("KeyS");
  expect(DEFAULT_BINDINGS.fly.left).toBe("KeyA");
  expect(DEFAULT_BINDINGS.fly.right).toBe("KeyD");
  expect(DEFAULT_BINDINGS.fly.up).toBe("KeyE");
  expect(DEFAULT_BINDINGS.fly.down).toBe("KeyQ");
  expect(DEFAULT_BINDINGS.fly.boost).toBe("ShiftLeft");
  expect(DEFAULT_BINDINGS.version).toBe(1);
});

test("validateBindings accepts the defaults", () => {
  expect(validateBindings(DEFAULT_BINDINGS)).toBe(true);
});

test("validateBindings rejects missing fields", () => {
  expect(validateBindings(null)).toBe(false);
  expect(validateBindings({})).toBe(false);
  expect(validateBindings({ mouse: {}, fly: {} })).toBe(false);
  expect(validateBindings({ version: 1, mouse: { left: "orbit" }, fly: DEFAULT_BINDINGS.fly })).toBe(false);
});

test("validateBindings rejects unknown mouse actions", () => {
  const bad = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  bad.mouse.left = "teleport";
  expect(validateBindings(bad)).toBe(false);
});

test("validateBindings rejects wrong version", () => {
  const bad = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  bad.version = 999;
  expect(validateBindings(bad)).toBe(false);
});

test("loadBindings returns defaults when storage is empty", () => {
  expect(loadBindings()).toEqual(DEFAULT_BINDINGS);
});

test("loadBindings returns defaults when JSON is malformed", () => {
  localStorage.setItem(STORAGE_KEY, "{not json");
  expect(loadBindings()).toEqual(DEFAULT_BINDINGS);
});

test("saveBindings + loadBindings round-trip", () => {
  const custom = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  custom.mouse.left = "fly";
  custom.mouse.right = "orbit";
  custom.fly.boost = "ControlLeft";
  saveBindings(custom);
  expect(loadBindings()).toEqual(custom);
});
