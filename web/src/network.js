/**
 * network.js -- WebSocket communication for the Pylos frontend.
 *
 * Exports:
 *   connect(onMessage)     -- establish WS, auto-reconnect
 *   send(msg)              -- send JSON over the WS
 *   fetchCheckpoints()     -- GET /checkpoints, return parsed JSON
 */

let ws = null;
let _onMessage = null;
let _reconnectTimer = null;

const RECONNECT_DELAY_MS = 2000;

/**
 * Build the WebSocket URL from the current page location.
 */
function _wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/game`;
}

/**
 * Connect to the game WebSocket.
 * @param {Function} onMessage - callback receiving parsed JSON messages
 */
export function connect(onMessage) {
  _onMessage = onMessage;
  _open();
}

function _open() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  ws = new WebSocket(_wsUrl());

  ws.addEventListener("open", () => {
    console.log("[ws] connected");
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
  });

  ws.addEventListener("message", (event) => {
    try {
      const data = JSON.parse(event.data);
      if (_onMessage) _onMessage(data);
    } catch (err) {
      console.error("[ws] failed to parse message:", err);
    }
  });

  ws.addEventListener("close", () => {
    console.log("[ws] disconnected, will reconnect...");
    _scheduleReconnect();
  });

  ws.addEventListener("error", (err) => {
    console.error("[ws] error:", err);
    // The close event will fire next, triggering reconnect
  });
}

function _scheduleReconnect() {
  if (_reconnectTimer) return;
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    _open();
  }, RECONNECT_DELAY_MS);
}

/**
 * Send a JSON message over the WebSocket.
 * @param {object} msg - message object (will be JSON-stringified)
 */
export function send(msg) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn("[ws] cannot send, not connected");
    return;
  }
  ws.send(JSON.stringify(msg));
}

/**
 * Fetch the checkpoint manifest from the server.
 * @returns {Promise<object>} parsed JSON (e.g. { checkpoints: [...] })
 */
export async function fetchCheckpoints() {
  try {
    const resp = await fetch("/checkpoints");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error("[network] fetchCheckpoints failed:", err);
    return { checkpoints: [] };
  }
}
