"""Metrics reporting for Pylos training pipeline.

Provides both file-based (backward-compatible) and HTTP-based metric reporting.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_progress_prev = {}  # path -> (step, time) for rolling speed


class MetricsReporter:
    """Encapsulates file-based and optional HTTP metric reporting.

    Args:
        ckpt_dir: Directory for writing training_progress.json and loss_history.jsonl.
        dashboard_url: Optional base URL for HTTP reporting (e.g. "http://localhost:8080").
                       Empty string or None disables HTTP reporting.
    """

    def __init__(self, ckpt_dir, dashboard_url=None):
        self.ckpt_dir = ckpt_dir
        self.dashboard_url = (dashboard_url or "").rstrip("/")
        self.progress_path = os.path.join(ckpt_dir, "training_progress.json")
        self.loss_history_path = os.path.join(ckpt_dir, "loss_history.jsonl")

    def _post_json(self, endpoint, payload):
        """POST JSON to dashboard. Logs warning on failure, never raises."""
        if not self.dashboard_url:
            return
        url = f"{self.dashboard_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning("Dashboard POST to %s failed: %s", url, e)

    def report_progress(self, step, total, value_loss, policy_loss, elapsed, eta,
                        status="training"):
        """Write training_progress.json and optionally POST to dashboard.

        Args:
            step: Current game/step number.
            total: Total games/steps planned.
            value_loss: Current value loss (or None).
            policy_loss: Current policy loss (or None).
            elapsed: Seconds elapsed since training start.
            eta: Estimated seconds remaining.
            status: Training status string (e.g. "training", "starting", "complete").
        """
        now = time.time()

        # Rolling speed calculation
        prev = _progress_prev.get(self.progress_path)
        if prev and now > prev[1] and step > prev[0]:
            recent_rate = (step - prev[0]) / (now - prev[1])
        else:
            recent_rate = step / elapsed if elapsed > 0 else 0
        _progress_prev[self.progress_path] = (step, now)

        progress = {
            "status": status,
            "current_game": step,
            "total_games": total,
            "percent": round(step / total * 100, 1) if total > 0 else 0,
            "value_loss": round(value_loss, 6) if value_loss is not None else None,
            "policy_loss": round(policy_loss, 6) if policy_loss is not None else None,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "games_per_second": round(recent_rate, 3),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        self._post_json("/api/metrics", progress)

    def report_loss(self, step, value_loss, policy_loss):
        """Append to loss_history.jsonl and optionally POST to dashboard.

        Args:
            step: Current game/step number.
            value_loss: Value head loss.
            policy_loss: Policy head loss.
        """
        entry = {
            "step": step,
            "value_loss": round(value_loss, 6),
            "policy_loss": round(policy_loss, 6),
        }

        with open(self.loss_history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._post_json("/api/metrics", {
            "type": "loss",
            **entry,
        })

    def report_checkpoint(self, filename, step, elo=None, label=None, win_rate=None):
        """POST checkpoint info to dashboard (no file writing).

        Args:
            filename: Checkpoint filename.
            step: Training step at checkpoint.
            elo: Optional ELO rating.
            label: Optional quality label.
            win_rate: Optional win rate vs previous checkpoint.
        """
        payload = {
            "filename": filename,
            "step": step,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if elo is not None:
            payload["elo"] = elo
        if label is not None:
            payload["label"] = label
        if win_rate is not None:
            payload["win_rate"] = win_rate

        self._post_json("/api/checkpoint", payload)
