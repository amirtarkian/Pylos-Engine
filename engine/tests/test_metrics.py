"""Tests for MetricsReporter."""

import json
import os
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import MetricsReporter


class TestFileReporting:
    """Tests for file-based metric reporting (backward compat)."""

    def test_report_progress_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d)
            r.report_progress(50, 100, 0.5, 0.3, 120.0, 120.0)

            path = os.path.join(d, "training_progress.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)

            assert data["current_game"] == 50
            assert data["total_games"] == 100
            assert data["percent"] == 50.0
            assert data["value_loss"] == 0.5
            assert data["policy_loss"] == 0.3
            assert data["status"] == "training"
            assert "timestamp" in data

    def test_report_progress_with_none_losses(self):
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d)
            r.report_progress(0, 100, None, None, 0.0, 0.0, status="starting")

            with open(os.path.join(d, "training_progress.json")) as f:
                data = json.load(f)

            assert data["value_loss"] is None
            assert data["policy_loss"] is None
            assert data["status"] == "starting"

    def test_report_loss_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d)
            r.report_loss(10, 0.5, 0.3)
            r.report_loss(20, 0.4, 0.2)

            path = os.path.join(d, "loss_history.jsonl")
            assert os.path.exists(path)

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            entry1 = json.loads(lines[0])
            assert entry1["step"] == 10
            assert entry1["value_loss"] == 0.5
            assert entry1["policy_loss"] == 0.3

            entry2 = json.loads(lines[1])
            assert entry2["step"] == 20

    def test_report_checkpoint_no_crash_without_url(self):
        """report_checkpoint should be a no-op when no dashboard URL."""
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d)
            # Should not raise
            r.report_checkpoint("checkpoint_00050.pth", 50, elo=1200, label="Strong")


class TestHTTPReporting:
    """Tests for HTTP-based metric reporting."""

    def test_handles_connection_refused(self):
        """HTTP failure should log warning, not crash."""
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d, dashboard_url="http://127.0.0.1:19999")
            # These should all complete without raising
            r.report_progress(10, 100, 0.5, 0.3, 60.0, 60.0)
            r.report_loss(10, 0.5, 0.3)
            r.report_checkpoint("ckpt.pth", 10)

    def test_posts_to_dashboard(self):
        """Verify correct POST payload is sent."""
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                received.append({"path": self.path, "body": body})
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress server logs in test output

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        try:
            with tempfile.TemporaryDirectory() as d:
                r = MetricsReporter(d, dashboard_url=f"http://127.0.0.1:{port}")

                r.report_progress(50, 100, 0.5, 0.3, 60.0, 60.0)
                r.report_loss(10, 0.5, 0.3)
                r.report_checkpoint("ckpt.pth", 10, elo=1200, label="Strong",
                                    win_rate=0.65)

            assert len(received) == 3

            # report_progress -> /api/metrics
            assert received[0]["path"] == "/api/metrics"
            assert received[0]["body"]["current_game"] == 50

            # report_loss -> /api/metrics
            assert received[1]["path"] == "/api/metrics"
            assert received[1]["body"]["type"] == "loss"
            assert received[1]["body"]["step"] == 10

            # report_checkpoint -> /api/checkpoint
            assert received[2]["path"] == "/api/checkpoint"
            assert received[2]["body"]["filename"] == "ckpt.pth"
            assert received[2]["body"]["elo"] == 1200
            assert received[2]["body"]["win_rate"] == 0.65
        finally:
            server.shutdown()

    def test_empty_url_disables_http(self):
        """Empty dashboard_url should not attempt any HTTP calls."""
        with tempfile.TemporaryDirectory() as d:
            r = MetricsReporter(d, dashboard_url="")
            # If these tried to POST, they'd fail since there's no server
            r.report_progress(10, 100, 0.5, 0.3, 60.0, 60.0)
            r.report_loss(10, 0.5, 0.3)
            r.report_checkpoint("ckpt.pth", 10)
            # Should write files but no HTTP
            assert os.path.exists(os.path.join(d, "training_progress.json"))
