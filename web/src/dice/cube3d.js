/**
 * cube3d.js -- Three.js view of the dice Sudoku cube.
 *
 * Exports:
 *   initCube(container, geometry)   -- set up scene, start render loop
 *   setState(values)                -- rebuild stickers from 54 values
 *   animateMove(action, ms)         -- turn a layer (returns a Promise)
 *   updateHighlights(values)        -- tint duplicates / completed faces
 *   setHoverFacelet(index|null)     -- outline one sticker (editor hover)
 *   setOnFaceletClick(cb)           -- click a sticker -> cb(faceletIndex)
 *
 * Cubies are persistent 3D objects that really rotate, so the pip
 * orientation of 2s, 3s and 6s is tracked exactly like on the real cube.
 * The logical state (values per facelet) is owned by main.js and updated
 * with the server's move permutations; this module only mirrors it.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { drawPips } from "./pips.js";

const BG = 0x1a1a2e;
const PIP_WHITE = new THREE.Color(0xf4f4f0);
const PIP_DUP = new THREE.Color(0xff8a5c);    // value repeated on this face
const PIP_DONE = new THREE.Color(0x7ee2a8);   // face already holds 1-9

let renderer, scene, camera, controls, raycaster;
let geo = null;            // geometry from /api/geometry
let cubies = [];           // THREE.Group per cubie
let stickers = [];         // { mesh, value }
let faceletLookup = new Map();
let textures = {};
let busy = Promise.resolve();
let hoverMesh = null;
let onFaceletClick = null;
let moveLabel = null;

export function initCube(container, geometry) {
  geo = geometry;
  geo.positions.forEach((p, i) => {
    faceletLookup.set(key(p, geo.normals[i]), i);
  });

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);

  camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(5.4, 5.0, 8.6);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 5;
  controls.maxDistance = 16;

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key1 = new THREE.DirectionalLight(0xffffff, 1.4);
  key1.position.set(5, 8, 6);
  scene.add(key1);
  const fill = new THREE.DirectionalLight(0x8899ff, 0.5);
  fill.position.set(-6, -3, -4);
  scene.add(fill);

  for (let v = 1; v <= 9; v++) textures[v] = makeTexture(v, false);
  textures.center = makeTexture(5, true);

  hoverMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.44, 0.5, 4, 1, Math.PI / 4),
    new THREE.MeshBasicMaterial({ color: 0x5ab0ff, side: THREE.DoubleSide })
  );
  hoverMesh.visible = false;
  scene.add(hoverMesh);

  raycaster = new THREE.Raycaster();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
    const hit = pick(e);
    if (hit !== null && onFaceletClick) onFaceletClick(hit);
  });

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  moveLabel = document.createElement("div");
  moveLabel.className = "move-flash";
  container.appendChild(moveLabel);

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function key(p, n) {
  return `${p.join(",")}|${n.join(",")}`;
}

function makeTexture(value, center) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  drawPips(c.getContext("2d"), value, 256, { bg: "#0d0d0f", pip: "#ffffff", center });
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function stickerUp(normal) {
  // Matches the 2D net: U is read with B at the top, D with F at the top.
  if (normal[1] === 1) return new THREE.Vector3(0, 0, -1);
  if (normal[1] === -1) return new THREE.Vector3(0, 0, 1);
  return new THREE.Vector3(0, 1, 0);
}

/** Rebuild all cubies/stickers from a 54-value array. */
export function setState(values) {
  for (const c of cubies) scene.remove(c);
  cubies = [];
  stickers = [];

  const body = new THREE.BoxGeometry(0.98, 0.98, 0.98);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x050506, roughness: 0.55 });
  const plane = new THREE.PlaneGeometry(0.9, 0.9);
  const byPos = new Map();

  geo.positions.forEach((p, i) => {
    const k = p.join(",");
    if (!byPos.has(k)) {
      const g = new THREE.Group();
      g.position.set(p[0], p[1], p[2]);
      g.add(new THREE.Mesh(body, bodyMat));
      scene.add(g);
      byPos.set(k, g);
      cubies.push(g);
    }
    const n = geo.normals[i];
    const isCenter = Math.abs(p[0]) + Math.abs(p[1]) + Math.abs(p[2]) === 1;
    const mat = new THREE.MeshStandardMaterial({
      map: isCenter ? textures.center : textures[values[i]],
      roughness: 0.4,
    });
    const mesh = new THREE.Mesh(plane, mat);
    const z = new THREE.Vector3(...n);
    const y = stickerUp(n);
    const x = new THREE.Vector3().crossVectors(y, z);
    mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
    mesh.position.copy(z.clone().multiplyScalar(0.495));
    mesh.userData.value = values[i];
    byPos.get(k).add(mesh);
    stickers.push(mesh);
  });

  // Centers never move relative to each other, but 5s look the same any way up
  updateHighlights(values);
}

/** Which facelet index a sticker currently occupies. */
function faceletOf(mesh) {
  const wp = new THREE.Vector3();
  mesh.getWorldPosition(wp);
  const wn = new THREE.Vector3(0, 0, 1).applyQuaternion(mesh.getWorldQuaternion(new THREE.Quaternion()));
  const n = [Math.round(wn.x), Math.round(wn.y), Math.round(wn.z)];
  const p = [Math.round(wp.x - 0.495 * n[0]), Math.round(wp.y - 0.495 * n[1]), Math.round(wp.z - 0.495 * n[2])];
  return faceletLookup.get(key(p, n));
}

/** Tint pips: orange where a value repeats on its face, green on finished faces. */
export function updateHighlights(values) {
  const faceDone = [];
  const dup = new Array(54).fill(false);
  for (let f = 0; f < 6; f++) {
    const seen = {};
    for (let j = 0; j < 9; j++) seen[values[f * 9 + j]] = (seen[values[f * 9 + j]] || 0) + 1;
    faceDone.push(Object.keys(seen).length === 9);
    for (let j = 0; j < 9; j++) if (seen[values[f * 9 + j]] > 1) dup[f * 9 + j] = true;
  }
  for (const s of stickers) {
    const i = faceletOf(s);
    if (i === undefined) continue;
    const f = Math.floor(i / 9);
    s.material.color.copy(faceDone[f] ? PIP_DONE : dup[i] ? PIP_DUP : PIP_WHITE);
  }
}

/** Animate one quarter turn. action 0-5 = U R F D L B clockwise, 6-11 = primes. */
export function animateMove(action, ms = 300) {
  busy = busy.then(() => new Promise((resolve) => {
    const face = geo.faces[action % 6];
    const n = geo.normals[geo.faces.indexOf(face) * 9 + 4];
    const axis = new THREE.Vector3(...n);
    const angle = (action < 6 ? -1 : 1) * Math.PI / 2;
    const layer = cubies.filter((c) => Math.round(c.position.dot(axis)) === 1);
    const pivot = new THREE.Object3D();
    scene.add(pivot);
    layer.forEach((c) => pivot.attach(c));

    if (moveLabel && ms >= 120) {
      moveLabel.textContent = geo.actions[action];
      moveLabel.classList.remove("show");
      void moveLabel.offsetWidth;
      moveLabel.classList.add("show");
    }

    const start = performance.now();
    const finish = () => {
      pivot.quaternion.setFromAxisAngle(axis, angle);
      pivot.updateMatrixWorld(true);
      layer.forEach((c) => {
        scene.attach(c);
        c.position.set(Math.round(c.position.x), Math.round(c.position.y), Math.round(c.position.z));
        snapRotation(c);
      });
      scene.remove(pivot);
      resolve();
    };
    if (ms <= 0) return finish();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / ms);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      pivot.quaternion.setFromAxisAngle(axis, angle * e);
      if (t < 1) requestAnimationFrame(tick);
      else finish();
    };
    requestAnimationFrame(tick);
  }));
  return busy;
}

function snapRotation(obj) {
  const m = new THREE.Matrix4().makeRotationFromQuaternion(obj.quaternion);
  const e = m.elements.map((v) => Math.round(v));
  m.fromArray(e);
  obj.quaternion.setFromRotationMatrix(m);
}

export function setHoverFacelet(index) {
  if (index === null || index === undefined) {
    hoverMesh.visible = false;
    return;
  }
  const p = geo.positions[index];
  const n = geo.normals[index];
  const z = new THREE.Vector3(...n);
  hoverMesh.position.set(p[0] + n[0] * 0.505, p[1] + n[1] * 0.505, p[2] + n[2] * 0.505);
  hoverMesh.lookAt(hoverMesh.position.clone().add(z));
  hoverMesh.visible = true;
}

export function setOnFaceletClick(cb) {
  onFaceletClick = cb;
}

function pick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(stickers, false);
  if (!hits.length) return null;
  const i = faceletOf(hits[0].object);
  return i === undefined ? null : i;
}

/** Point the camera so a given face is toward the viewer (editor helper). */
export function lookAtFace(faceIndex) {
  const n = geo.normals[faceIndex * 9 + 4];
  const dir = new THREE.Vector3(n[0] * 1.0 + 0.45, n[1] * 1.0 + 0.4, n[2] * 1.0 + 0.6).normalize();
  if (n[1] !== 0) dir.set(0.25, n[1], n[1] > 0 ? 0.55 : 0.55).normalize();
  camera.position.copy(dir.multiplyScalar(camera.position.length()));
  controls.update();
}
