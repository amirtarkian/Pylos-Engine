/**
 * camera-controls.js -- Orbit + Fly controls facade.
 *
 * Usage:
 *   const cc = createCameraControls(camera, domElement, { bindings });
 *   cc.update();         // call once per render frame
 *   cc.setBindings(b);   // swap bindings at runtime
 *   cc.dispose();
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DEFAULT_BINDINGS } from "./camera-bindings.js";

const ORBIT_ACTION_MAP = {
  orbit: THREE.MOUSE.ROTATE,
  pan: THREE.MOUSE.PAN,
  // "fly" and "none" -> not handled by OrbitControls; set to null below.
};

const FLY_SPEED_BASE = 4.0;       // world units per second
const FLY_SPEED_BOOST = 3.0;      // multiplier when boost held
const FLY_LOOK_SENS = 0.0025;     // radians per pixel of pointer movement
const FLY_WHEEL_SPEED_STEP = 0.2; // multiplicative
const FLY_EXIT_DISTANCE = 5;      // target placed this far in front on exit

export function createCameraControls(camera, dom, opts = {}) {
  let bindings = opts.bindings ? structuredClone(opts.bindings) : structuredClone(DEFAULT_BINDINGS);

  const orbit = new OrbitControls(camera, dom);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.minDistance = 4;
  orbit.maxDistance = 15;
  orbit.maxPolarAngle = Math.PI * 0.44;

  // Fly state
  let flying = false;
  let flySpeed = 1.0; // multiplier on FLY_SPEED_BASE, tuned by wheel
  const heldKeys = new Set();
  let yaw = 0, pitch = 0;
  let lastTime = performance.now();

  function applyBindingsToOrbit() {
    const m = bindings.mouse;
    orbit.mouseButtons = {
      LEFT:   ORBIT_ACTION_MAP[m.left]   ?? null,
      MIDDLE: ORBIT_ACTION_MAP[m.middle] ?? null,
      RIGHT:  ORBIT_ACTION_MAP[m.right]  ?? null,
    };
  }
  applyBindingsToOrbit();

  function flyButtonIndex() {
    const m = bindings.mouse;
    if (m.left === "fly") return 0;
    if (m.middle === "fly") return 1;
    if (m.right === "fly") return 2;
    return -1;
  }

  function syncYawPitchFromCamera() {
    const e = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
    yaw = e.y;
    pitch = e.x;
  }

  function applyYawPitchToCamera() {
    const maxPitch = Math.PI / 2 - 0.001;
    pitch = Math.max(-maxPitch, Math.min(maxPitch, pitch));
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));
    camera.quaternion.copy(q);
  }

  function enterFly() {
    if (flying) return;
    flying = true;
    orbit.enabled = false;
    syncYawPitchFromCamera();
    if (dom.requestPointerLock) {
      try { dom.requestPointerLock(); } catch (_) { /* fall back to free mouse */ }
    }
  }

  function exitFly() {
    if (!flying) return;
    flying = false;
    if (document.pointerLockElement === dom && document.exitPointerLock) {
      document.exitPointerLock();
    }
    // Place orbit target a fixed distance in front of the camera.
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    orbit.target.copy(camera.position).addScaledVector(forward, FLY_EXIT_DISTANCE);
    orbit.enabled = true;
    orbit.update();
    heldKeys.clear();
  }

  function onPointerDown(ev) {
    if (ev.button === flyButtonIndex()) {
      ev.preventDefault();
      enterFly();
    }
  }

  function onPointerUp(ev) {
    if (ev.button === flyButtonIndex() && flying) {
      ev.preventDefault();
      exitFly();
    }
  }

  function onPointerMove(ev) {
    if (!flying) return;
    const dx = ev.movementX || 0;
    const dy = ev.movementY || 0;
    yaw   -= dx * FLY_LOOK_SENS;
    pitch -= dy * FLY_LOOK_SENS;
    applyYawPitchToCamera();
  }

  function onKeyDown(ev) {
    if (!flying) return;
    heldKeys.add(ev.code);
  }

  function onKeyUp(ev) {
    heldKeys.delete(ev.code);
  }

  function onWheel(ev) {
    if (!flying) return;
    ev.preventDefault();
    const dir = Math.sign(ev.deltaY);
    if (dir > 0) flySpeed = Math.max(0.1, flySpeed - FLY_WHEEL_SPEED_STEP);
    else if (dir < 0) flySpeed = Math.min(10, flySpeed + FLY_WHEEL_SPEED_STEP);
  }

  function onContextMenu(ev) {
    // If right-mouse is bound to anything (fly/orbit/pan/none), suppress the browser menu.
    if (bindings.mouse.right !== "none") ev.preventDefault();
  }

  function onPointerLockChange() {
    // If the user pressed Esc to release the lock while still holding the mouse, exit fly.
    if (flying && document.pointerLockElement !== dom) {
      exitFly();
    }
  }

  dom.addEventListener("pointerdown", onPointerDown);
  dom.addEventListener("pointerup", onPointerUp);
  dom.addEventListener("pointermove", onPointerMove);
  dom.addEventListener("wheel", onWheel, { passive: false });
  dom.addEventListener("contextmenu", onContextMenu);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  document.addEventListener("pointerlockchange", onPointerLockChange);

  function update() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastTime) / 1000);
    lastTime = now;

    if (flying) {
      const k = bindings.fly;
      const boost = heldKeys.has(k.boost) ? FLY_SPEED_BOOST : 1.0;
      const speed = FLY_SPEED_BASE * flySpeed * boost * dt;

      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      const worldUp = new THREE.Vector3(0, 1, 0);

      if (heldKeys.has(k.forward)) camera.position.addScaledVector(forward, speed);
      if (heldKeys.has(k.back))    camera.position.addScaledVector(forward, -speed);
      if (heldKeys.has(k.left))    camera.position.addScaledVector(right, -speed);
      if (heldKeys.has(k.right))   camera.position.addScaledVector(right, speed);
      if (heldKeys.has(k.up))      camera.position.addScaledVector(worldUp, speed);
      if (heldKeys.has(k.down))    camera.position.addScaledVector(worldUp, -speed);
    } else {
      orbit.update();
    }
  }

  function setBindings(nextBindings) {
    if (flying) exitFly();
    bindings = structuredClone(nextBindings);
    applyBindingsToOrbit();
  }

  function dispose() {
    if (flying) exitFly();
    orbit.dispose();
    dom.removeEventListener("pointerdown", onPointerDown);
    dom.removeEventListener("pointerup", onPointerUp);
    dom.removeEventListener("pointermove", onPointerMove);
    dom.removeEventListener("wheel", onWheel);
    dom.removeEventListener("contextmenu", onContextMenu);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    document.removeEventListener("pointerlockchange", onPointerLockChange);
  }

  return {
    update,
    setBindings,
    dispose,
    get orbit() { return orbit; },
    get isFlying() { return flying; },
  };
}
