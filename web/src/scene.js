/**
 * scene.js -- Three.js 3D scene for the Pylos board game.
 *
 * Exports:
 *   init(container)                   -- set up scene, start render loop
 *   setCallbacks(posClickCb, sphereClickCb) -- register interaction handlers
 *   updateBoard(board, legalMoves)    -- sync 3D objects with game state
 *   highlightSphere(level, row, col)  -- green glow on a sphere (raise selection)
 *   clearHighlights()                 -- remove all sphere highlights
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LEVEL_SIZES, boardToWorld } from "./game.js";

// ── Internal state ───────────────────────────────────────────────
let renderer, scene, camera, controls;
let raycaster, mouse;
let positionMarkers = [];   // { mesh, level, row, col }
let sphereMeshes = [];      // { mesh, level, row, col, player }
let animatingMeshes = [];   // { mesh, startY, endY, startTime, duration }
let onPositionClick = null;
let onSphereClick = null;

// Shared geometries & materials (created once)
let sphereGeoWhite, sphereGeoBlack;
let matWhite, matBlack;
let markerGeo, markerMatDefault, markerMatLegal;

// ── Constants ────────────────────────────────────────────────────
const BG_COLOR = 0x1a1a2e;
const SPHERE_RADIUS = 0.45;
const DROP_DURATION = 500;   // ms
const DROP_HEIGHT = 5;       // units above final position

// ── Easing ───────────────────────────────────────────────────────

/** Bounce-out easing for drop animation. */
function easeOutBounce(t) {
  if (t < 1 / 2.75) {
    return 7.5625 * t * t;
  } else if (t < 2 / 2.75) {
    t -= 1.5 / 2.75;
    return 7.5625 * t * t + 0.75;
  } else if (t < 2.5 / 2.75) {
    t -= 2.25 / 2.75;
    return 7.5625 * t * t + 0.9375;
  } else {
    t -= 2.625 / 2.75;
    return 7.5625 * t * t + 0.984375;
  }
}

// ── Init ─────────────────────────────────────────────────────────

export function init(container) {
  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(BG_COLOR);
  scene.fog = new THREE.Fog(BG_COLOR, 12, 28);

  // Camera
  camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );
  camera.position.set(6, 8, 6);
  camera.lookAt(0, 1.5, 0);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 1.5, 0);
  controls.minDistance = 4;
  controls.maxDistance = 15;
  controls.maxPolarAngle = Math.PI * 0.44; // ~80 degrees
  controls.update();

  // Raycaster
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // ── Lighting ────────────────────────────────────────────────
  const ambient = new THREE.AmbientLight(0x404060, 0.6);
  scene.add(ambient);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(5, 10, 5);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.near = 0.5;
  dirLight.shadow.camera.far = 30;
  dirLight.shadow.camera.left = -8;
  dirLight.shadow.camera.right = 8;
  dirLight.shadow.camera.top = 8;
  dirLight.shadow.camera.bottom = -8;
  dirLight.shadow.bias = -0.001;
  scene.add(dirLight);

  const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
  fillLight.position.set(-3, 5, -3);
  scene.add(fillLight);

  // ── Board base ──────────────────────────────────────────────
  const boardGeo = new THREE.BoxGeometry(7, 0.3, 7);
  const boardMat = new THREE.MeshStandardMaterial({
    color: 0x8b6914,
    roughness: 0.85,
    metalness: 0.05,
  });
  const boardMesh = new THREE.Mesh(boardGeo, boardMat);
  boardMesh.position.set(0, -0.15, 0);
  boardMesh.receiveShadow = true;
  scene.add(boardMesh);

  // Decorative edge strip
  const edgeGeo = new THREE.BoxGeometry(7.2, 0.06, 7.2);
  const edgeMat = new THREE.MeshStandardMaterial({
    color: 0x6b5010,
    roughness: 0.9,
    metalness: 0.0,
  });
  const edgeMesh = new THREE.Mesh(edgeGeo, edgeMat);
  edgeMesh.position.set(0, 0.01, 0);
  edgeMesh.receiveShadow = true;
  scene.add(edgeMesh);

  // ── Shared geometries / materials ───────────────────────────
  sphereGeoWhite = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 64);
  sphereGeoBlack = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 64);

  matWhite = new THREE.MeshStandardMaterial({
    color: 0xf5f5f0,
    roughness: 0.3,
    metalness: 0.1,
  });
  matBlack = new THREE.MeshStandardMaterial({
    color: 0x2a2a2a,
    roughness: 0.6,
    metalness: 0.2,
  });

  markerGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.02, 32);
  markerMatDefault = new THREE.MeshStandardMaterial({
    color: 0x666666,
    transparent: true,
    opacity: 0.15,
    roughness: 0.8,
  });
  markerMatLegal = new THREE.MeshStandardMaterial({
    color: 0x4a6cf7,
    transparent: true,
    opacity: 0.45,
    roughness: 0.4,
    emissive: 0x4a6cf7,
    emissiveIntensity: 0.3,
  });

  // ── Position markers (all 30 positions) ─────────────────────
  for (let level = 0; level < 4; level++) {
    const size = LEVEL_SIZES[level];
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const pos = boardToWorld(level, row, col);
        const marker = new THREE.Mesh(markerGeo, markerMatDefault.clone());
        marker.position.set(pos.x, pos.y - SPHERE_RADIUS + 0.01, pos.z);
        marker.receiveShadow = true;
        marker.userData = { type: "position", level, row, col };
        scene.add(marker);
        positionMarkers.push({ mesh: marker, level, row, col });
      }
    }
  }

  // ── Events ──────────────────────────────────────────────────
  renderer.domElement.addEventListener("pointermove", onPointerMove, false);
  renderer.domElement.addEventListener("pointerdown", onPointerDown, false);

  // Resize
  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  // Start render loop
  _animate();
}

// ── Callbacks ────────────────────────────────────────────────────

export function setCallbacks(posClickCb, sphereClickCb) {
  onPositionClick = posClickCb;
  onSphereClick = sphereClickCb;
}

// ── Board sync ───────────────────────────────────────────────────

/**
 * Synchronise the 3D scene with the current game board.
 *
 * @param {Array} board       - 4-level 2D array (null | "white" | "black")
 * @param {Array} legalMoves  - Array of legal move objects from the server
 */
export function updateBoard(board, legalMoves) {
  // -- Build a set of currently occupied positions from the board --
  const occupied = new Map(); // "level,row,col" -> player
  for (let level = 0; level < 4; level++) {
    const size = LEVEL_SIZES[level];
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (board[level][r][c]) {
          occupied.set(`${level},${r},${c}`, board[level][r][c]);
        }
      }
    }
  }

  // -- Remove meshes for pieces no longer on the board --
  const toRemove = [];
  for (let i = sphereMeshes.length - 1; i >= 0; i--) {
    const entry = sphereMeshes[i];
    const key = `${entry.level},${entry.row},${entry.col}`;
    const occupant = occupied.get(key);
    if (!occupant || occupant !== entry.player) {
      scene.remove(entry.mesh);
      entry.mesh.geometry.dispose();
      entry.mesh.material.dispose();
      sphereMeshes.splice(i, 1);
    }
  }

  // -- Build a set of existing mesh positions --
  const existingKeys = new Set();
  for (const entry of sphereMeshes) {
    existingKeys.add(`${entry.level},${entry.row},${entry.col}`);
  }

  // -- Add meshes for new pieces --
  for (const [key, player] of occupied) {
    if (existingKeys.has(key)) continue;
    const [level, row, col] = key.split(",").map(Number);
    _addSphere(level, row, col, player, true);
  }

  // -- Update position markers: highlight legal placements --
  const legalPlaceKeys = new Set();
  if (legalMoves) {
    for (const move of legalMoves) {
      if (move.type === "place") {
        legalPlaceKeys.add(`${move.level},${move.row},${move.col}`);
      }
    }
  }

  for (const pm of positionMarkers) {
    const key = `${pm.level},${pm.row},${pm.col}`;
    const isLegal = legalPlaceKeys.has(key);
    const isOccupied = occupied.has(key);

    if (isLegal && !isOccupied) {
      pm.mesh.material.color.setHex(0x4a6cf7);
      pm.mesh.material.opacity = 0.45;
      pm.mesh.material.emissive = new THREE.Color(0x4a6cf7);
      pm.mesh.material.emissiveIntensity = 0.3;
    } else {
      pm.mesh.material.color.setHex(0x666666);
      pm.mesh.material.opacity = isOccupied ? 0.0 : 0.15;
      pm.mesh.material.emissive = new THREE.Color(0x000000);
      pm.mesh.material.emissiveIntensity = 0;
    }
  }
}

// ── Highlight / clear ────────────────────────────────────────────

/**
 * Add a green emissive glow to the sphere at (level, row, col).
 */
export function highlightSphere(level, row, col) {
  for (const entry of sphereMeshes) {
    if (entry.level === level && entry.row === row && entry.col === col) {
      entry.mesh.material.emissive = new THREE.Color(0x44cc44);
      entry.mesh.material.emissiveIntensity = 0.5;
      break;
    }
  }
}

/**
 * Remove all sphere highlights.
 */
export function clearHighlights() {
  for (const entry of sphereMeshes) {
    entry.mesh.material.emissive = new THREE.Color(0x000000);
    entry.mesh.material.emissiveIntensity = 0;
  }
}

// ── Internal helpers ─────────────────────────────────────────────

/**
 * Create a sphere mesh and add it to the scene.
 * @param {boolean} animate - If true, play drop animation.
 */
function _addSphere(level, row, col, player, animate) {
  const geo = player === "white" ? sphereGeoWhite : sphereGeoBlack;
  const baseMat = player === "white" ? matWhite : matBlack;
  const mat = baseMat.clone();
  const mesh = new THREE.Mesh(geo, mat);

  const pos = boardToWorld(level, row, col);
  mesh.position.set(pos.x, pos.y, pos.z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = { type: "sphere", level, row, col, player };

  scene.add(mesh);
  sphereMeshes.push({ mesh, level, row, col, player });

  if (animate) {
    const startY = pos.y + DROP_HEIGHT;
    mesh.position.y = startY;
    animatingMeshes.push({
      mesh,
      startY,
      endY: pos.y,
      startTime: performance.now(),
      duration: DROP_DURATION,
    });
  }
}

// ── Pointer events ───────────────────────────────────────────────

function _getIntersects(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const targets = [
    ...positionMarkers.map((pm) => pm.mesh),
    ...sphereMeshes.map((sm) => sm.mesh),
  ];
  return raycaster.intersectObjects(targets, false);
}

function onPointerMove(event) {
  const intersects = _getIntersects(event);
  if (intersects.length > 0) {
    const ud = intersects[0].object.userData;
    if (ud.type === "position" || ud.type === "sphere") {
      renderer.domElement.style.cursor = "pointer";
      return;
    }
  }
  renderer.domElement.style.cursor = "default";
}

function onPointerDown(event) {
  // Ignore right-click / middle-click
  if (event.button !== 0) return;

  const intersects = _getIntersects(event);
  if (intersects.length === 0) return;

  const obj = intersects[0].object;
  const ud = obj.userData;

  if (ud.type === "sphere" && onSphereClick) {
    onSphereClick(ud.level, ud.row, ud.col, ud.player);
  } else if (ud.type === "position" && onPositionClick) {
    onPositionClick(ud.level, ud.row, ud.col);
  }
}

// ── Render loop ──────────────────────────────────────────────────

function _animate() {
  requestAnimationFrame(_animate);

  const now = performance.now();

  // Process drop animations
  for (let i = animatingMeshes.length - 1; i >= 0; i--) {
    const anim = animatingMeshes[i];
    const elapsed = now - anim.startTime;
    const t = Math.min(elapsed / anim.duration, 1);
    const eased = easeOutBounce(t);

    anim.mesh.position.y = anim.startY + (anim.endY - anim.startY) * eased;

    if (t >= 1) {
      anim.mesh.position.y = anim.endY;
      animatingMeshes.splice(i, 1);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}
