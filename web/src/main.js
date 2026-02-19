/**
 * main.js -- Entry point for the Pylos web frontend.
 *
 * Wires together scene, network, game state, and DOM UI.
 */

import { createEmptyBoard, updateBoardFromState } from "./game.js";
import { init as initScene, setCallbacks, updateBoard, highlightSphere, clearHighlights, flashFormation, flashAndRemove } from "./scene.js";
import { connect, send, fetchCheckpoints, fetchTrainingStatus, fetchLossHistory } from "./network.js";

// ── State ────────────────────────────────────────────────────────
let board = createEmptyBoard();
let legalMoves = [];
let currentTurn = "white";
let gameOver = false;
let moveCount = 0;

// Raise selection state
let selectedSphere = null; // { level, row, col, player } or null

// Removal phase state
let removalPhase = false;
let removablePieces = []; // [{level, row, col}, ...]

// AI removal animation state
let pendingAiRemovals = null; // [{level, row, col}, ...] or null
let aiRemovalAnimating = false;
const DROP_ANIM_WAIT = 600; // ms to wait for placement drop animation before flashing removals

// Move history navigation state
let boardSnapshots = []; // [{board: serverBoard, turn, reserves}, ...]
let viewIndex = null;    // null = live (latest), number = viewing history

// Track pending human move for logging to history
let pendingHumanMove = null; // {action, playerColor} or null

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
const btnDoneRemovingEl = document.getElementById("btn-done-removing");

// Dual AI selectors
const groupAiWhiteEl = document.getElementById("group-ai-white");
const groupAiBlackEl = document.getElementById("group-ai-black");
const selAiWhiteEl = document.getElementById("sel-ai-white");
const selAiBlackEl = document.getElementById("sel-ai-black");

// Search iterations (AI strength)
const groupSearchItersEl = document.getElementById("group-search-iters");
const selSearchItersEl = document.getElementById("sel-search-iters");

// Move navigation
const navFirstEl = document.getElementById("nav-first");
const navPrevEl = document.getElementById("nav-prev");
const navPauseEl = document.getElementById("nav-pause");
const navNextEl = document.getElementById("nav-next");
const navLastEl = document.getElementById("nav-last");

// Move position indicator
const movePositionEl = document.getElementById("move-position");

// Pause state
let gamePaused = false;

// Speed slider
const groupSpeedEl = document.getElementById("group-speed");
const sliderSpeedEl = document.getElementById("slider-speed");
const speedLabelEl = document.getElementById("speed-label");

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

// ── Move history helper (chess-style paired rows) ────────────────

let turnNumber = 0;       // current turn pair (1, 2, 3, ...)
let currentMoveRow = null; // DOM element for current row being built

function addMoveToHistory(playerColor, moveDesc, thinkingMs) {
  moveCount++;
  const snapshotIdx = moveCount; // 1-based index into boardSnapshots

  let timeHtml = "";
  if (thinkingMs !== undefined) {
    timeHtml = `<span class="move-time">${(thinkingMs / 1000).toFixed(1)}s</span>`;
  }

  if (playerColor === "white") {
    // Create a new paired row
    turnNumber++;
    currentMoveRow = document.createElement("div");
    currentMoveRow.className = "move-row";

    const numCell = document.createElement("span");
    numCell.className = "move-num-cell";
    numCell.textContent = `${turnNumber}.`;

    const whiteCell = document.createElement("span");
    whiteCell.className = "move-cell white-move";
    whiteCell.innerHTML = moveDesc + timeHtml;
    whiteCell.dataset.idx = snapshotIdx;
    whiteCell.addEventListener("click", () => navigateToMove(parseInt(whiteCell.dataset.idx)));

    const blackCell = document.createElement("span");
    blackCell.className = "move-cell placeholder";
    blackCell.textContent = "\u2026"; // ellipsis

    currentMoveRow.appendChild(numCell);
    currentMoveRow.appendChild(whiteCell);
    currentMoveRow.appendChild(blackCell);
    moveHistoryEl.appendChild(currentMoveRow);
  } else {
    // Fill the black cell of the current row
    if (currentMoveRow) {
      const blackCell = currentMoveRow.querySelector(".move-cell.placeholder");
      if (blackCell) {
        blackCell.className = "move-cell black-move";
        blackCell.innerHTML = moveDesc + timeHtml;
        blackCell.dataset.idx = snapshotIdx;
        blackCell.addEventListener("click", () => navigateToMove(parseInt(blackCell.dataset.idx)));
      }
    }
  }

  // Auto-scroll only when viewing live
  if (viewIndex === null) {
    moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
  }
  updateMovePosition();
}

function clearMoveHistory() {
  moveHistoryEl.innerHTML = "";
  moveCount = 0;
  turnNumber = 0;
  currentMoveRow = null;
  boardSnapshots = [];
  viewIndex = null;
  updateNavButtons();
  updateMovePosition();
}

// ── Move navigation helpers ──────────────────────────────────────

function saveSnapshot(serverBoard, turn, reserves) {
  boardSnapshots.push({ board: serverBoard, turn, reserves });
}

function navigateToMove(idx) {
  if (idx < 0 || idx >= boardSnapshots.length) return;
  viewIndex = idx;
  showSnapshot(idx);
  updateNavButtons();
  highlightMoveEntry(idx);
  updateMovePosition();
}

function navigateToLive() {
  viewIndex = null;
  if (boardSnapshots.length > 0) {
    showSnapshot(boardSnapshots.length - 1);
  }
  updateNavButtons();
  highlightMoveEntry(null);
  updateMovePosition();
}

function showSnapshot(idx) {
  const snap = boardSnapshots[idx];
  if (!snap) return;
  const tempBoard = updateBoardFromState(createEmptyBoard(), snap.board);
  updateBoard(tempBoard, []); // No legal moves when viewing history
  updateTurnUI(snap.turn);
  updateReservesUI(snap.reserves);
  clearHighlights();

  const isLive = viewIndex === null;
  if (!isLive) {
    setStatus(`Viewing move ${idx} of ${boardSnapshots.length - 1}`, "info");
  }
}

function updateNavButtons() {
  const total = boardSnapshots.length;
  const isLive = viewIndex === null;
  const pos = isLive ? total - 1 : viewIndex;

  navFirstEl.disabled = total <= 1 || pos <= 0;
  navPrevEl.disabled = total <= 1 || pos <= 0;
  // Enable "next" for stepping when paused in AI vs AI at live position,
  // or when navigating through history (not at the latest snapshot yet)
  const canStep = isLive && selModeEl.value === "ai_vs_ai" && gamePaused && !gameOver;
  const canAdvanceHistory = !isLive && pos < total - 1;
  navNextEl.disabled = !canStep && !canAdvanceHistory;
  navLastEl.disabled = isLive;

  if (isLive) {
    navLastEl.classList.add("live-active");
  } else {
    navLastEl.classList.remove("live-active");
  }

  updateMovePosition();
}

function highlightMoveEntry(idx) {
  const cells = moveHistoryEl.querySelectorAll(".move-cell:not(.placeholder)");
  cells.forEach(cell => {
    const cellIdx = parseInt(cell.dataset.idx);
    if (idx !== null && cellIdx === idx) {
      cell.classList.add("active");
      cell.scrollIntoView({ block: "nearest" });
    } else {
      cell.classList.remove("active");
    }
  });
  // If live, clear all highlights and auto-scroll
  if (idx === null) {
    cells.forEach(cell => cell.classList.remove("active"));
    moveHistoryEl.scrollTop = moveHistoryEl.scrollHeight;
  }
}

function isViewingHistory() {
  return viewIndex !== null;
}

function formatMoveDesc(action) {
  if (action.type === "place") {
    return `L${action.level}(${action.row},${action.col})`;
  } else if (action.type === "raise") {
    const s = action.src;
    const d = action.dst;
    return `L${s.level}(${s.row},${s.col})\u2192L${d.level}(${d.row},${d.col})`;
  }
  return "?";
}

function updateMovePosition() {
  if (!movePositionEl) return;
  const total = boardSnapshots.length - 1; // -1 because index 0 is initial state
  if (total <= 0) {
    movePositionEl.textContent = "";
    return;
  }
  const pos = viewIndex === null ? total : viewIndex;
  movePositionEl.textContent = `Move ${pos} / ${total}`;
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
    groupAiWhiteEl.classList.add("hidden");
    groupAiBlackEl.classList.add("hidden");
    groupSpeedEl.classList.add("hidden");
    groupSearchItersEl.classList.remove("hidden");
  } else if (mode === "ai_vs_ai") {
    groupDifficultyEl.classList.add("hidden");
    groupColorEl.classList.add("hidden");
    groupAiWhiteEl.classList.remove("hidden");
    groupAiBlackEl.classList.remove("hidden");
    groupSpeedEl.classList.remove("hidden");
    groupSearchItersEl.classList.remove("hidden");
  } else {
    // human_vs_human
    groupDifficultyEl.classList.add("hidden");
    groupColorEl.classList.add("hidden");
    groupAiWhiteEl.classList.add("hidden");
    groupAiBlackEl.classList.add("hidden");
    groupSpeedEl.classList.add("hidden");
    groupSearchItersEl.classList.add("hidden");
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
  if (isViewingHistory()) return; // No interaction when viewing history
  if (removalPhase) return; // During removal, only sphere clicks matter

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
      const actionObj = {
        type: "raise",
        src: {
          level: selectedSphere.level,
          row: selectedSphere.row,
          col: selectedSphere.col,
        },
        dst: { level, row, col },
        action: raiseMove.action,
      };
      pendingHumanMove = { action: actionObj, playerColor: currentTurn };
      send({ type: "move", action: actionObj });
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
    const actionObj = {
      type: "place",
      level,
      row,
      col,
      action: placeMove.action,
    };
    pendingHumanMove = { action: actionObj, playerColor: currentTurn };
    send({ type: "move", action: actionObj });
    setStatus("Move sent...", "info", 2000);
  }
}

function onSphereClick(level, row, col, player) {
  if (gameOver) return;
  if (isViewingHistory()) return; // No interaction when viewing history

  // Removal phase: clicking a highlighted removable piece removes it
  if (removalPhase) {
    const isRemovable = removablePieces.some(
      (p) => p.level === level && p.row === row && p.col === col
    );
    if (isRemovable) {
      send({ type: "remove", level, row, col });
      setStatus("Removing piece...", "info", 2000);
    }
    return;
  }

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
      removalPhase = false;
      removablePieces = [];
      if (btnDoneRemovingEl) btnDoneRemovingEl.classList.add("hidden");

      // Save board snapshot for navigation
      saveSnapshot(msg.board, msg.turn, msg.reserves);

      // Log pending human move to history
      if (pendingHumanMove) {
        const desc = formatMoveDesc(pendingHumanMove.action);
        addMoveToHistory(pendingHumanMove.playerColor, desc);
        pendingHumanMove = null;
      }

      // If AI just removed pieces, animate the removal before updating board
      if (pendingAiRemovals && pendingAiRemovals.length > 0 && !aiRemovalAnimating) {
        const removals = pendingAiRemovals;
        pendingAiRemovals = null;
        aiRemovalAnimating = true;

        // Determine AI color for putting removed pieces back temporarily
        const aiColor =
          selModeEl.value === "human_vs_ai"
            ? selColorEl.value === "white" ? "black" : "white"
            : currentTurn;

        // Build a temp board with removed pieces added back (to show placement + pieces still there)
        const tempBoard = updateBoardFromState(board, msg.board);
        for (const r of removals) {
          tempBoard[r.level][r.row][r.col] = aiColor;
        }

        // Update scene with temp board (shows AI placement drop + removed pieces still present)
        updateBoard(tempBoard, []);
        updateTurnUI(msg.turn);
        updateReservesUI(msg.reserves);
        selectedSphere = null;
        clearHighlights();

        // After a short delay for the placement drop to land, flash removed pieces green
        setTimeout(() => {
          flashAndRemove(removals, 1200, 600).then(() => {
            aiRemovalAnimating = false;
            // Now sync to the real board state (pieces already gone)
            board = updateBoardFromState(board, msg.board);
            legalMoves = msg.legal_moves || [];
            // Only update to live board if not viewing history
            if (!isViewingHistory()) {
              updateBoard(board, legalMoves);
            }
            updateNavButtons();

            const mode = selModeEl.value;
            if (mode === "human_vs_ai" && msg.turn !== selColorEl.value) {
              setStatus("AI thinking...", "info");
            } else if (mode === "ai_vs_ai") {
              setStatus(`${msg.turn}'s turn (AI)`, "info");
            } else {
              setStatus(`${msg.turn}'s turn`, "info", 3000);
            }
          });
        }, DROP_ANIM_WAIT);
        break;
      }

      pendingAiRemovals = null;
      board = updateBoardFromState(board, msg.board);
      legalMoves = msg.legal_moves || [];

      // Only update 3D board if viewing live (not browsing history)
      if (!isViewingHistory()) {
        updateBoard(board, legalMoves);
        updateTurnUI(msg.turn);
        updateReservesUI(msg.reserves);
      }
      updateNavButtons();

      // Clear raise selection on new state
      selectedSphere = null;
      clearHighlights();

      const mode = selModeEl.value;
      if (!isViewingHistory()) {
        if (mode === "human_vs_ai" && msg.turn !== selColorEl.value) {
          setStatus("AI thinking...", "info");
        } else if (mode === "ai_vs_ai") {
          setStatus(`${msg.turn}'s turn (AI)`, "info");
        } else {
          setStatus(`${msg.turn}'s turn`, "info", 3000);
        }
      }
      break;
    }

    case "removal_phase": {
      removablePieces = msg.removable || [];
      const formation = msg.formation || [];

      board = updateBoardFromState(board, msg.board);
      legalMoves = []; // No normal moves during removal
      updateBoard(board, []);
      updateReservesUI(msg.reserves);

      // Clear raise selection
      selectedSphere = null;
      clearHighlights();

      // Flash the completed formation in orange first, then highlight removable pieces green
      if (formation.length > 0 && msg.removed_so_far === 0) {
        setStatus(`${msg.player} completed a pattern!`, "info");
        flashFormation(formation, 800).then(() => {
          removalPhase = true;
          clearHighlights();
          for (const p of removablePieces) {
            highlightSphere(p.level, p.row, p.col);
          }
          const remaining = msg.max_removals - msg.removed_so_far;
          setStatus(
            `${msg.player}: Remove up to ${remaining} piece${remaining > 1 ? "s" : ""}, or click Done`,
            "info"
          );
          if (btnDoneRemovingEl) btnDoneRemovingEl.classList.remove("hidden");
        });
      } else {
        // Subsequent removal (after first piece removed) — skip the flash
        removalPhase = true;
        for (const p of removablePieces) {
          highlightSphere(p.level, p.row, p.col);
        }
        const remaining = msg.max_removals - msg.removed_so_far;
        setStatus(
          `${msg.player}: Remove up to ${remaining} piece${remaining > 1 ? "s" : ""}, or click Done`,
          "info"
        );
        if (btnDoneRemovingEl) btnDoneRemovingEl.classList.remove("hidden");
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

      let fullDesc = desc;
      if (msg.removed && msg.removed.length > 0) {
        const removedStr = msg.removed
          .map((p) => `L${p.level}(${p.row},${p.col})`)
          .join(",");
        fullDesc += ` \u00d7${removedStr}`;
        // Store removals so the next state update can animate them
        pendingAiRemovals = msg.removed;
      }
      addMoveToHistory(aiColor, fullDesc, msg.thinking_time_ms);

      if (msg.removed && msg.removed.length > 0) {
        setStatus("AI formed a pattern — removing pieces...", "info");
      }
      break;
    }

    case "game_over": {
      gameOver = true;
      const reasonText =
        msg.reason === "apex_placed"
          ? "Apex placed!"
          : msg.reason === "no_legal_moves"
          ? "No legal moves!"
          : msg.reason === "repetition"
          ? "Draw by repetition"
          : msg.reason === "move_limit"
          ? "Draw by move limit"
          : msg.reason;
      if (msg.winner === "draw") {
        setStatus(`Game over: ${reasonText}`, "info");
      } else {
        setStatus(`Game over: ${msg.winner} wins! (${reasonText})`, "success");
      }
      break;
    }

    case "paused": {
      gamePaused = msg.paused;
      navPauseEl.innerHTML = gamePaused ? "&#9654;" : "&#9646;&#9646;";
      navPauseEl.title = gamePaused ? "Resume" : "Pause";
      updateNavButtons();
      if (gamePaused) {
        setStatus("Paused — use ▶ to step one move", "info");
      } else {
        setStatus("Resumed", "info", 2000);
      }
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
  removalPhase = false;
  removablePieces = [];
  pendingAiRemovals = null;
  aiRemovalAnimating = false;
  boardSnapshots = [];
  viewIndex = null;
  pendingHumanMove = null;
  gamePaused = false;
  navPauseEl.innerHTML = "&#9646;&#9646;";
  navPauseEl.title = "Pause";
  if (btnDoneRemovingEl) btnDoneRemovingEl.classList.add("hidden");
  selectedSphere = null;
  clearHighlights();
  updateNavButtons();

  const payload = { type: "new_game", mode };

  if (mode === "human_vs_ai") {
    const checkpoint = selDifficultyEl.value;
    if (checkpoint) payload.checkpoint = checkpoint;
    payload.human_color = selColorEl.value;
    payload.search_iterations = parseInt(selSearchItersEl.value);
  } else if (mode === "ai_vs_ai") {
    const ckptWhite = selAiWhiteEl.value;
    const ckptBlack = selAiBlackEl.value;
    if (ckptWhite) payload.checkpoint_white = ckptWhite;
    if (ckptBlack) payload.checkpoint_black = ckptBlack;
    payload.delay_ms = parseInt(sliderSpeedEl.value);
    payload.search_iterations = parseInt(selSearchItersEl.value);
  }

  send(payload);
  setStatus("Starting new game...", "info", 2000);
}

// ── Training Dashboard ───────────────────────────────────────────

const checkpointListEl = document.getElementById("checkpoint-list");

// Per-run DOM elements and loss histories
const trainRuns = {};
for (const run of ["v1", "v2", "v3", "v4", "v5"]) {
  trainRuns[run] = {
    badge:       document.getElementById(`train-status-badge-${run}`),
    progressTxt: document.getElementById(`train-progress-text-${run}`),
    percent:     document.getElementById(`train-percent-${run}`),
    progressBar: document.getElementById(`train-progress-bar-${run}`),
    vloss:       document.getElementById(`train-vloss-${run}`),
    ploss:       document.getElementById(`train-ploss-${run}`),
    speed:       document.getElementById(`train-speed-${run}`),
    eta:         document.getElementById(`train-eta-${run}`),
    canvas:      document.getElementById(`loss-chart-${run}`),
    lossHistory: { value: [], policy: [] },
    lastStep: -1, // track last appended step to avoid duplicates from polling
  };
}

let lastCheckpointCount = 0;

// Past trainings tab switching
const pastTabs = document.querySelectorAll(".past-tab");
const pastPanels = document.querySelectorAll("#past-trainings .train-run-panel");

pastTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    const run = tab.dataset.run;
    pastTabs.forEach(t => t.classList.toggle("active", t === tab));
    pastPanels.forEach(p => p.classList.toggle("active", p.id === `train-run-${run}`));
    drawLossChart(run);
  });
});

// Past trainings collapse/expand
const btnPastTrainings = document.getElementById("btn-past-trainings");
const pastTrainingsEl = document.getElementById("past-trainings");
if (btnPastTrainings && pastTrainingsEl) {
  btnPastTrainings.addEventListener("click", () => {
    const isHidden = pastTrainingsEl.style.display === "none";
    pastTrainingsEl.style.display = isHidden ? "flex" : "none";
    pastTrainingsEl.classList.toggle("hidden", !isHidden);
    btnPastTrainings.textContent = isHidden ? "Past Trainings ▲" : "Past Trainings ▼";
  });
}

function formatTime(seconds) {
  if (seconds == null || seconds <= 0) return "—";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function updateRunDashboard(run, data) {
  const els = trainRuns[run];
  if (!els || !els.badge) return;

  const status = data.status || "idle";
  els.badge.textContent = status.toUpperCase();
  els.badge.className = "train-status " +
    (status === "training" || status === "starting" ? "training" :
     status === "complete" ? "complete" : "idle");

  const current = data.current_game || 0;
  const total = data.total_games || 0;
  const pct = data.percent || 0;
  els.progressTxt.textContent = `${current.toLocaleString()} / ${total.toLocaleString()} games`;
  els.percent.textContent = `${pct}%`;
  els.progressBar.style.width = `${pct}%`;

  els.vloss.textContent = data.value_loss != null ? data.value_loss.toFixed(4) : "—";
  els.ploss.textContent = data.policy_loss != null ? data.policy_loss.toFixed(4) : "—";
  els.speed.textContent = data.games_per_second != null ? `${data.games_per_second.toFixed(2)}/s` : "—";
  els.eta.textContent = formatTime(data.eta_seconds);

  // Only append new data points (avoid duplicates from polling same step)
  const step = data.current_game || 0;
  if (data.value_loss != null && step > els.lastStep) {
    els.lastStep = step;
    els.lossHistory.value.push(data.value_loss);
    els.lossHistory.policy.push(data.policy_loss);
    drawLossChart(run);
  }

  // Update tab badge color to reflect run status
  const tab = document.querySelector(`.train-tab[data-run="${run}"]`);
  if (tab) {
    tab.textContent = `${run.toUpperCase()} ${status === "training" ? "●" : status === "complete" ? "✓" : ""}`.trim();
  }
}

function drawLossChart(run) {
  const els = trainRuns[run];
  if (!els || !els.canvas) return;
  const canvas = els.canvas;
  const history = els.lossHistory;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 4, right: 4, bottom: 4, left: 4 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  if (history.value.length < 2) return;

  const allVals = [...history.value, ...history.policy];
  const maxVal = Math.max(...allVals, 0.001);
  const minVal = Math.min(...allVals, 0);

  function drawLine(data, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (i / (data.length - 1)) * cw;
      const y = pad.top + ch - ((data[i] - minVal) / (maxVal - minVal || 1)) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawLine(history.value, "#4a6cf7");
  drawLine(history.policy, "#6a3de8");

  ctx.font = "9px system-ui";
  ctx.fillStyle = "#4a6cf7";
  ctx.fillText("value", w - 70, 12);
  ctx.fillStyle = "#6a3de8";
  ctx.fillText("policy", w - 35, 12);
}

function updateCheckpointList(checkpoints) {
  checkpointListEl.innerHTML = "";
  if (!checkpoints || checkpoints.length === 0) {
    checkpointListEl.innerHTML = '<div style="font-size:11px;color:#555a70;padding:4px;">No checkpoints yet</div>';
    return;
  }

  const sorted = [...checkpoints].reverse();
  for (const cp of sorted) {
    const entry = document.createElement("div");
    entry.className = "ckpt-entry";

    const label = cp.label || `Step ${cp.step}`;
    const stat = cp.elo != null ? `ELO ${cp.elo}` :
                 cp.win_rate_vs_random != null ? `${Math.round(cp.win_rate_vs_random * 100)}%` : "...";
    const ver = cp.version || "v1";

    entry.innerHTML = `
      <span class="ckpt-label">${ver.toUpperCase()} ${label} <span style="color:#555a70;">#${cp.step}</span></span>
      <span class="ckpt-wr">${stat}</span>
      <button class="ckpt-play-btn" data-file="${cp.file}">Play</button>
    `;

    entry.querySelector(".ckpt-play-btn").addEventListener("click", (e) => {
      const file = e.target.dataset.file;
      selModeEl.value = "human_vs_ai";
      updateModeVisibility();
      for (const opt of selDifficultyEl.options) {
        if (opt.value === file) {
          selDifficultyEl.value = file;
          break;
        }
      }
      startNewGame();
    });

    checkpointListEl.appendChild(entry);
  }
}

function _populateSelector(selectEl, checkpoints) {
  const prevValue = selectEl.value;
  selectEl.innerHTML = "";

  if (checkpoints.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No checkpoints yet";
    selectEl.appendChild(opt);
    return;
  }

  for (const cp of checkpoints) {
    const opt = document.createElement("option");
    opt.value = cp.file || cp;
    if (typeof cp === "string") {
      opt.textContent = cp;
    } else {
      const wr = cp.win_rate_vs_random != null ? `${Math.round(cp.win_rate_vs_random * 100)}%` : "?";
      const stepK = cp.step != null ? `${(cp.step / 1000).toFixed(1)}k` : "?";
      const version = cp.version || ((cp.file || "").includes("v2") ? "v2" : "v1");
      opt.textContent = `${version}-${stepK} — ${wr} WR (${cp.label || "?"})`;
    }
    selectEl.appendChild(opt);
  }

  const prevExists = [...selectEl.options].some(o => o.value === prevValue);
  if (prevExists) {
    selectEl.value = prevValue;
  } else {
    selectEl.value = selectEl.options[selectEl.options.length - 1].value;
  }
}

function refreshDifficultySelector(checkpoints) {
  _populateSelector(selDifficultyEl, checkpoints);
  _populateSelector(selAiWhiteEl, checkpoints);
  _populateSelector(selAiBlackEl, checkpoints);

  // Default Black AI to the first (earliest/weakest) checkpoint
  // so the two selectors don't start on the same model
  if (checkpoints.length > 1 && selAiBlackEl.value === selAiWhiteEl.value) {
    selAiBlackEl.value = selAiBlackEl.options[0].value;
  }
}

// ── ELO Chart ─────────────────────────────────────────────────────

const eloChartCanvas = document.getElementById("elo-chart");

function winRateToElo(wr) {
  // Clamp to avoid log(0) or log(infinity)
  const clamped = Math.max(0.01, Math.min(0.99, wr));
  return 1000 - 400 * Math.log10((1 - clamped) / clamped);
}

function drawEloChart(checkpoints) {
  const canvas = eloChartCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 16, right: 8, bottom: 20, left: 36 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  // Split by version — only include checkpoints with a known ELO
  const seriesMap = {};
  for (const cp of checkpoints) {
    if (cp.elo == null && cp.win_rate_vs_random == null) continue;
    const elo = cp.elo != null ? cp.elo : winRateToElo(cp.win_rate_vs_random);
    const ver = cp.version || "v1";
    if (!seriesMap[ver]) seriesMap[ver] = [];
    seriesMap[ver].push({ step: cp.step, elo });
  }

  const allPoints = Object.values(seriesMap).flat();
  if (allPoints.length === 0) return;
  const maxStep = Math.max(...allPoints.map(p => p.step), 1);
  const allElos = allPoints.map(p => p.elo);
  let minElo = Math.min(...allElos);
  let maxElo = Math.max(...allElos);

  // Ensure at least some range so chart isn't a flat line
  if (maxElo - minElo < 50) {
    const mid = (maxElo + minElo) / 2;
    minElo = mid - 50;
    maxElo = mid + 50;
  }

  // Add 10% padding
  const eloRange = maxElo - minElo;
  minElo -= eloRange * 0.1;
  maxElo += eloRange * 0.1;

  function toX(step) { return pad.left + (step / maxStep) * cw; }
  function toY(elo) { return pad.top + ch - ((elo - minElo) / (maxElo - minElo)) * ch; }

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  const eloTicks = 5;
  for (let i = 0; i <= eloTicks; i++) {
    const elo = minElo + (i / eloTicks) * (maxElo - minElo);
    const y = toY(elo);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Y axis labels
    ctx.fillStyle = "#555a70";
    ctx.font = "9px system-ui";
    ctx.textAlign = "right";
    ctx.fillText(Math.round(elo), pad.left - 4, y + 3);
  }

  // X axis labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#555a70";
  const stepTicks = Math.min(5, Math.floor(maxStep / 1000));
  for (let i = 0; i <= Math.max(stepTicks, 1); i++) {
    const step = Math.round((i / Math.max(stepTicks, 1)) * maxStep);
    const x = toX(step);
    const label = step >= 1000 ? `${(step / 1000).toFixed(0)}k` : `${step}`;
    ctx.fillText(label, x, h - 4);
  }

  // Draw lines
  function drawSeries(points, color) {
    if (points.length < 2) {
      // Draw single point as a dot
      if (points.length === 1) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(toX(points[0].step), toY(points[0].elo), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      return;
    }
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (let i = 0; i < points.length; i++) {
      const x = toX(points[i].step);
      const y = toY(points[i].elo);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw endpoint dot
    const last = points[points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(toX(last.step), toY(last.elo), 3, 0, Math.PI * 2);
    ctx.fill();
  }

  const seriesColors = { v1: "#4a6cf7", v2: "#e8563d", v3: "#51cf66", v4: "#f0c040", v5: "#a855f7" };
  for (const [ver, points] of Object.entries(seriesMap)) {
    drawSeries(points, seriesColors[ver] || "#aaa");
  }
}

async function loadLossHistory() {
  const history = await fetchLossHistory();
  for (const [run, entries] of Object.entries(history)) {
    if (!trainRuns[run]) continue;
    const els = trainRuns[run];
    // Replace the in-memory history with persisted data
    els.lossHistory.value = entries.map(e => e.value_loss);
    els.lossHistory.policy = entries.map(e => e.policy_loss);
    // Set lastStep so polling doesn't re-add existing points
    if (entries.length > 0) {
      els.lastStep = entries[entries.length - 1].step;
    }
    drawLossChart(run);
  }
}

async function pollTraining() {
  const [statusData, ckptData] = await Promise.all([
    fetchTrainingStatus(),
    fetchCheckpoints(),
  ]);

  // Update all runs dynamically
  for (const run of Object.keys(statusData)) {
    if (trainRuns[run]) updateRunDashboard(run, statusData[run]);
  }

  const checkpoints = ckptData.checkpoints || [];
  if (checkpoints.length !== lastCheckpointCount) {
    lastCheckpointCount = checkpoints.length;
    refreshDifficultySelector(checkpoints);
  }

  updateCheckpointList(checkpoints);
  drawEloChart(checkpoints);
}

function startTrainingPoll() {
  // Load full loss history first, then start polling for new data
  loadLossHistory().then(() => {
    pollTraining();
    setInterval(pollTraining, 3000);
  });
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
  btnDoneRemovingEl.addEventListener("click", () => {
    if (removalPhase) {
      send({ type: "skip_removal" });
      setStatus("Skipping remaining removals...", "info", 2000);
    }
  });

  // Move navigation buttons
  navFirstEl.addEventListener("click", () => navigateToMove(0));
  navPrevEl.addEventListener("click", () => {
    if (viewIndex === null) {
      navigateToMove(boardSnapshots.length - 2);
    } else if (viewIndex > 0) {
      navigateToMove(viewIndex - 1);
    }
  });
  navNextEl.addEventListener("click", () => {
    if (viewIndex !== null) {
      if (viewIndex >= boardSnapshots.length - 1) {
        navigateToLive();
      } else {
        navigateToMove(viewIndex + 1);
      }
    } else if (selModeEl.value === "ai_vs_ai" && gamePaused && !gameOver) {
      // Step mode: play one AI move while staying paused
      send({ type: "step" });
      setStatus("Stepping...", "info", 2000);
    }
  });
  navLastEl.addEventListener("click", () => navigateToLive());

  // Speed slider
  sliderSpeedEl.addEventListener("input", () => {
    const ms = parseInt(sliderSpeedEl.value);
    speedLabelEl.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
    send({ type: "set_delay", delay_ms: ms });
  });

  // Play/pause button
  navPauseEl.addEventListener("click", () => {
    if (gamePaused) {
      send({ type: "resume" });
    } else {
      send({ type: "pause" });
    }
  });

  // Keyboard shortcuts for navigation
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      navPrevEl.click();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      navNextEl.click();
    } else if (e.key === "Home") {
      e.preventDefault();
      navFirstEl.click();
    } else if (e.key === "End") {
      e.preventDefault();
      navLastEl.click();
    } else if (e.key === " ") {
      e.preventDefault();
      navPauseEl.click();
    }
  });

  // Training panel toggle
  const btnToggleTraining = document.getElementById("btn-toggle-training");
  const trainingPanel = document.getElementById("training-panel");
  if (btnToggleTraining && trainingPanel) {
    btnToggleTraining.addEventListener("click", () => {
      const isHidden = trainingPanel.classList.toggle("hidden");
      btnToggleTraining.textContent = isHidden ? "Training ▼" : "Training ▲";
      // Shift toggle button down when panel is open so it doesn't overlap
      btnToggleTraining.style.top = isHidden ? "16px" : "";
      btnToggleTraining.style.display = isHidden ? "" : "none";
    });
    // Also add a close button behavior: clicking the panel header hides it
    const panelHeader = trainingPanel.querySelector("h2");
    if (panelHeader) {
      panelHeader.style.cursor = "pointer";
      panelHeader.title = "Click to collapse";
      panelHeader.addEventListener("click", () => {
        trainingPanel.classList.add("hidden");
        btnToggleTraining.textContent = "Training ▼";
        btnToggleTraining.style.display = "";
      });
    }
  }

  // Initial mode visibility
  updateModeVisibility();

  // Start training dashboard polling (also loads checkpoints)
  startTrainingPoll();

  // Connect WebSocket
  connect(handleMessage);

  setStatus("Connected. Press New Game to start.", "info");
}

main();
