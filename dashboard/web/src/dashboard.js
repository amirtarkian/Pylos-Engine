// ── Chart.js shared config ───────────────────────────────────
const CHART_COLORS = {
  valueLoss: "rgba(74, 108, 247, 1)",
  valueLossFill: "rgba(74, 108, 247, 0.08)",
  policyLoss: "rgba(255, 159, 64, 1)",
  policyLossFill: "rgba(255, 159, 64, 0.08)",
  elo: "rgba(81, 207, 102, 1)",
  eloFill: "rgba(81, 207, 102, 0.08)",
  grid: "rgba(255, 255, 255, 0.06)",
  tick: "#7a82a0",
};

const SCALE_DEFAULTS = {
  grid: { color: CHART_COLORS.grid },
  ticks: { color: CHART_COLORS.tick, font: { size: 11 } },
};

Chart.defaults.color = CHART_COLORS.tick;

// ── DOM refs ─────────────────────────────────────────────────
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const progressFill = document.getElementById("progress-fill");
const progressPct = document.getElementById("progress-pct");
const statGps = document.getElementById("stat-gps");
const statElapsed = document.getElementById("stat-elapsed");
const statEta = document.getElementById("stat-eta");
const statStep = document.getElementById("stat-step");
const sseDot = document.getElementById("sse-dot");
const sseLabel = document.getElementById("sse-label");

// ── Loss chart ───────────────────────────────────────────────
const lossChart = new Chart(document.getElementById("loss-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Value Loss",
        data: [],
        borderColor: CHART_COLORS.valueLoss,
        backgroundColor: CHART_COLORS.valueLossFill,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 6,
        fill: true,
        tension: 0.3,
      },
      {
        label: "Policy Loss",
        data: [],
        borderColor: CHART_COLORS.policyLoss,
        backgroundColor: CHART_COLORS.policyLossFill,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 6,
        fill: true,
        tension: 0.3,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { boxWidth: 12, padding: 16, font: { size: 12 } },
      },
      tooltip: {
        backgroundColor: "rgba(20, 20, 45, 0.9)",
        borderColor: "rgba(255, 255, 255, 0.1)",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ...SCALE_DEFAULTS,
        title: { display: true, text: "Training Step", color: CHART_COLORS.tick },
      },
      y: {
        ...SCALE_DEFAULTS,
        title: { display: true, text: "Loss", color: CHART_COLORS.tick },
        beginAtZero: false,
      },
    },
  },
});

// ── ELO chart ────────────────────────────────────────────────
const eloChart = new Chart(document.getElementById("elo-chart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "ELO Rating",
        data: [],
        borderColor: CHART_COLORS.elo,
        backgroundColor: CHART_COLORS.eloFill,
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: CHART_COLORS.elo,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.3,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { boxWidth: 12, padding: 16, font: { size: 12 } },
      },
      tooltip: {
        backgroundColor: "rgba(20, 20, 45, 0.9)",
        borderColor: "rgba(255, 255, 255, 0.1)",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ...SCALE_DEFAULTS,
        title: { display: true, text: "Checkpoint Step", color: CHART_COLORS.tick },
      },
      y: {
        ...SCALE_DEFAULTS,
        title: { display: true, text: "ELO", color: CHART_COLORS.tick },
      },
    },
  },
});

// ── Helpers ──────────────────────────────────────────────────
function formatTime(seconds) {
  if (seconds == null || seconds < 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function setStatus(state, label) {
  statusBadge.dataset.state = state;
  statusText.textContent = label;
}

function updateProgress(metrics) {
  const current = metrics.current_game ?? metrics.step ?? 0;
  const total = metrics.total_games ?? metrics.total ?? 0;
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;

  progressFill.style.width = pct.toFixed(1) + "%";
  progressPct.textContent = pct.toFixed(1) + "%";

  // Status
  if (pct >= 100) {
    setStatus("complete", "Complete");
  } else if (current > 0) {
    setStatus("training", "Training");
  } else {
    setStatus("idle", "Idle");
  }

  // Stats
  const elapsed = metrics.elapsed ?? null;
  const eta = metrics.eta ?? null;
  const gps =
    elapsed && elapsed > 0 && current > 0
      ? (current / elapsed).toFixed(1)
      : "—";

  statGps.textContent = gps;
  statElapsed.textContent = formatTime(elapsed);
  statEta.textContent = formatTime(eta);
  statStep.textContent = current > 0 ? current.toLocaleString() : "—";
}

function appendLossPoint(step, valueLoss, policyLoss) {
  lossChart.data.labels.push(step);
  lossChart.data.datasets[0].data.push(valueLoss);
  lossChart.data.datasets[1].data.push(policyLoss);
}

function appendEloPoint(step, elo) {
  eloChart.data.labels.push(step);
  eloChart.data.datasets[0].data.push(elo);
}

// ── Initial data load ────────────────────────────────────────
async function loadInitialData() {
  try {
    const [metricsRes, lossRes, ckptRes] = await Promise.all([
      fetch("/api/metrics"),
      fetch("/api/loss-history"),
      fetch("/api/checkpoints"),
    ]);

    if (metricsRes.ok) {
      const metrics = await metricsRes.json();
      updateProgress(metrics);
    }

    if (lossRes.ok) {
      const losses = await lossRes.json();
      for (const entry of losses) {
        appendLossPoint(
          entry.step,
          entry.value_loss,
          entry.policy_loss
        );
      }
      lossChart.update("none");
    }

    if (ckptRes.ok) {
      const checkpoints = await ckptRes.json();
      for (const ckpt of checkpoints) {
        if (ckpt.elo != null) {
          appendEloPoint(ckpt.step, ckpt.elo);
        }
      }
      eloChart.update("none");
    }
  } catch {
    // Server not ready yet — SSE will pick up data when it connects
  }
}

// ── SSE client ───────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource("/api/events");

  es.onopen = () => {
    sseDot.classList.add("connected");
    sseLabel.textContent = "Connected";
  };

  es.onerror = () => {
    sseDot.classList.remove("connected");
    sseLabel.textContent = "Reconnecting…";
  };

  es.addEventListener("metrics", (e) => {
    const data = JSON.parse(e.data);
    updateProgress(data);

    if (data.value_loss != null && data.policy_loss != null) {
      const step = data.current_game ?? data.step ?? lossChart.data.labels.length;
      appendLossPoint(step, data.value_loss, data.policy_loss);
      lossChart.update("none");
    }
  });

  es.addEventListener("checkpoint", (e) => {
    const data = JSON.parse(e.data);
    if (data.elo != null) {
      appendEloPoint(data.step, data.elo);
      eloChart.update("none");
    }
  });
}

// ── Boot ─────────────────────────────────────────────────────
loadInitialData();
connectSSE();
