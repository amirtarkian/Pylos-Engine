/**
 * main.js -- Entry point for the Pylos web frontend.
 *
 * Wires together scene, network, game state, and DOM UI.
 */

import { createEmptyBoard, updateBoardFromState } from "./game.js";
import { init as initScene, setCallbacks, updateBoard, highlightSphere, clearHighlights } from "./scene.js";
import { connect, send, fetchCheckpoints } from "./network.js";

// ── State ────────────────────────────────────────────────────────
let board = createEmptyBoard();
let legalMoves = [];
let currentTurn = "white";
let gameOver = false;
let moveCount = 0;

// Raise selection state
let selectedSphere = null; // { level, row, col, player } or null

// ── DOM refs ─────────────────────────────────────────────────────
const whiteReservesEl = document.getElementById("white-reserves");
const blackReservesEl = document.getElementById("black-reserves");
const turnIndicatorEl = document.getElementById("turn-indicator");
const statusBarEl = document.getElementById("status-bar");
const moveHistoryEl = document.getElementById("move-history");
const selModeEl = document.getElementById("sel-mode");
const selDifficultyEl = document.getElementById("sel-difficulty");
const selColorEl = document.getElementById("sel-color");
const btnNewGameEl = document.getElementById("btn-new-game");
const groupDifficultyEl = document.getElementById("group-difficulty");
const groupColorEl = document.getElementById("group-color");

// ── Status bar helpers ───────────────────────────────────────────

let statusTimeout = null;

function setStatus(text, className, duration) {
  statusBarEl.textContent = text;
  statusBarEl.className = "glass";
  if (className) statusBarEl.classList.add(className);
  statusBarEl.style.opacity = "1";

  if (statusTimeout) clearTimeout(statusTimeout);
  if (duration) {
    statusTimeout = setTimeout(() => {
      statusBarEl.style.opacity = "0.6";
    }, duration);
  }
}

// ── Move history helper ──────────────────────────────────────────

function addMoveToHistory(playerColor, moveDesc, thinkingMs) {
  moveCount++;
  const entry = document.createElement("div");
  entry.className = "move-entry";

  let timeStr = "";
  if (thinkingMs !== undefined) {
    timeStr = ` <span style="color:#555a70;">(${(thinkingMs / 1000).toFixed(1)}s)</span>`;
  }

  entry.innerHTML =
    `<span class="move-num">${moveCount}.</span>` +
    `<span class="move-player ${playerColor}">${playerColor}</span> ` +
    `<span>${moveDesc}</span>${timeStr}`;

  moveHistoryEl.appendChild(entry);
  moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
}

function clearMoveHistory() {
  moveHistoryEl.innerHTML = "";
  moveCount = 0;
}

function formatMoveDesc(action) {
  if (action.type === "place") {
    return `place L${action.level} (${action.row},${action.col})`;
  } else if (action.type === "raise") {
    const s = action.src;
    const d = action.dst;
    return `raise L${s.level}(${s.row},${s.col}) -> L${d.level}(${d.row},${d.col})`;
  }
  return "?";
}

// ── UI state updates ─────────────────────────────────────────────

function updateTurnUI(turn) {
  currentTurn = turn;
  turnIndicatorEl.textContent = turn === "white" ? "WHITE'S TURN" : "BLACK'S TURN";
}

function updateReservesUI(reserves) {
  whiteReservesEl.textContent = reserves.white;
  blackReservesEl.textContent = reserves.black;
}

function updateModeVisibility() {
  const mode = selModeEl.value;
  if (mode === "human_vs_ai") {
    groupDifficultyEl.classList.remove("hidden");
    groupColorEl.classList.remove("hidden");
  } else if (mode === "ai_vs_ai") {
    groupDifficultyEl.classList.remove("hidden");
    groupColorEl.classList.add("hidden");
  } else {
    // human_vs_human
    groupDifficultyEl.classList.add("hidden");
    groupColorEl.classList.add("hidden");
  }
}

// ── Checkpoint loading ───────────────────────────────────────────

async function loadCheckpoints() {
  const data = await fetchCheckpoints();
  const checkpoints = data.checkpoints || [];

  selDifficultyEl.innerHTML = "";

  if (checkpoints.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No checkpoints available";
    selDifficultyEl.appendChild(opt);
    return;
  }

  for (const cp of checkpoints) {
    const opt = document.createElement("option");
    // Support both string entries and object entries { file, label }
    if (typeof cp === "string") {
      opt.value = cp;
      opt.textContent = cp.replace(/\.pth$/, "").replace(/_/g, " ");
    } else {
      opt.value = cp.file || cp.filename || cp;
      opt.textContent = cp.label || cp.name || opt.value;
    }
    selDifficultyEl.appendChild(opt);
  }
}

// ── Click handlers ───────────────────────────────────────────────

function onPositionClick(level, row, col) {
  if (gameOver) return;

  // If we have a sphere selected for raising, this is the destination
  if (selectedSphere) {
    // Find matching raise move
    const raiseMove = legalMoves.find(
      (m) =>
        m.type === "raise" &&
        m.src.level === selectedSphere.level &&
        m.src.row === selectedSphere.row &&
        m.src.col === selectedSphere.col &&
        m.dst.level === level &&
        m.dst.row === row &&
        m.dst.col === col
    );

    if (raiseMove) {
      send({
        type: "move",
        action: {
          type: "raise",
          src: {
            level: selectedSphere.level,
            row: selectedSphere.row,
            col: selectedSphere.col,
          },
          dst: { level, row, col },
          action: raiseMove.action,
        },
      });
      selectedSphere = null;
      clearHighlights();
      setStatus("Move sent...", "info", 2000);
      return;
    }

    // Destination wasn't a valid raise target -- clear selection and fall through
    selectedSphere = null;
    clearHighlights();
  }

  // Normal placement
  const placeMove = legalMoves.find(
    (m) =>
      m.type === "place" &&
      m.level === level &&
      m.row === row &&
      m.col === col
  );

  if (placeMove) {
    send({
      type: "move",
      action: {
        type: "place",
        level,
        row,
        col,
        action: placeMove.action,
      },
    });
    setStatus("Move sent...", "info", 2000);
  }
}

function onSphereClick(level, row, col, player) {
  if (gameOver) return;

  // Check if this sphere is a valid raise source
  const canRaise = legalMoves.some(
    (m) =>
      m.type === "raise" &&
      m.src.level === level &&
      m.src.row === row &&
      m.src.col === col
  );

  if (canRaise && player === currentTurn) {
    // Toggle selection
    if (
      selectedSphere &&
      selectedSphere.level === level &&
      selectedSphere.row === row &&
      selectedSphere.col === col
    ) {
      // Deselect
      selectedSphere = null;
      clearHighlights();
      setStatus("Selection cleared", "info", 1500);
    } else {
      selectedSphere = { level, row, col, player };
      clearHighlights();
      highlightSphere(level, row, col);
      setStatus("Select a destination to raise the sphere", "info");
    }
  } else {
    // Clicked a sphere that can't be raised -- try treating it as a position click
    // (for positions that already have a sphere on top, this is a no-op)
    selectedSphere = null;
    clearHighlights();
  }
}

// ── WebSocket message handler ────────────────────────────────────

function handleMessage(msg) {
  switch (msg.type) {
    case "state": {
      gameOver = false;
      board = updateBoardFromState(board, msg.board);
      legalMoves = msg.legal_moves || [];
      updateBoard(board, legalMoves);
      updateTurnUI(msg.turn);
      updateReservesUI(msg.reserves);

      // Clear raise selection on new state
      selectedSphere = null;
      clearHighlights();

      const mode = selModeEl.value;
      if (mode === "human_vs_ai" && msg.turn !== selColorEl.value) {
        setStatus("AI thinking...", "info");
      } else if (mode === "ai_vs_ai") {
        setStatus(`${msg.turn}'s turn (AI)`, "info");
      } else {
        setStatus(`${msg.turn}'s turn`, "info", 3000);
      }
      break;
    }

    case "ai_move": {
      const desc = formatMoveDesc(msg.action);
      // Determine AI color: in human_vs_ai it's the opposite of human color,
      // in ai_vs_ai we track by turn.
      const aiColor =
        selModeEl.value === "human_vs_ai"
          ? selColorEl.value === "white"
            ? "black"
            : "white"
          : currentTurn;
      addMoveToHistory(aiColor, desc, msg.thinking_time_ms);
      break;
    }

    case "game_over": {
      gameOver = true;
      const reason =
        msg.reason === "apex_placed"
          ? "Apex placed!"
          : msg.reason === "no_legal_moves"
          ? "No legal moves!"
          : msg.reason;
      setStatus(`Game over: ${msg.winner} wins! (${reason})`, "success");
      break;
    }

    case "error": {
      setStatus(`Error: ${msg.message}`, "error", 4000);
      break;
    }

    default:
      console.warn("[main] Unknown message type:", msg.type);
  }
}

// ── New Game ─────────────────────────────────────────────────────

function startNewGame() {
  const mode = selModeEl.value;
  clearMoveHistory();
  gameOver = false;
  selectedSphere = null;
  clearHighlights();

  const payload = { type: "new_game", mode };

  if (mode === "human_vs_ai") {
    const checkpoint = selDifficultyEl.value;
    if (checkpoint) payload.checkpoint = checkpoint;
    payload.human_color = selColorEl.value;
  } else if (mode === "ai_vs_ai") {
    const checkpoint = selDifficultyEl.value;
    if (checkpoint) payload.checkpoint = checkpoint;
    payload.delay_ms = 1500;
  }

  send(payload);
  setStatus("Starting new game...", "info", 2000);
}

// ── Initialization ───────────────────────────────────────────────

function main() {
  // Init 3D scene
  const container = document.getElementById("canvas-container");
  initScene(container);
  setCallbacks(onPositionClick, onSphereClick);

  // UI event listeners
  selModeEl.addEventListener("change", updateModeVisibility);
  btnNewGameEl.addEventListener("click", startNewGame);

  // Initial mode visibility
  updateModeVisibility();

  // Fetch checkpoints
  loadCheckpoints();

  // Connect WebSocket
  connect(handleMessage);

  setStatus("Connected. Press New Game to start.", "info");
}

main();
