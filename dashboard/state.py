"""In-memory training metrics store with thread safety."""

import asyncio
import threading
from typing import Any


class TrainingState:
    """Thread-safe store for training metrics, loss history, and checkpoints.

    Supports SSE broadcasting via asyncio queues.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._progress: dict[str, Any] = {}
        self._loss_history: list[dict[str, Any]] = []
        self._checkpoints: list[dict[str, Any]] = []
        self._subscribers: list[asyncio.Queue] = []

    # -- Reads (lock-protected copies) --

    def get_progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    def get_loss_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._loss_history)

    def get_checkpoints(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._checkpoints)

    # -- Writes --

    def update_progress(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._progress.update(data)
        self._broadcast("metrics", data)

    def add_loss(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._loss_history.append(entry)
        self._broadcast("metrics", entry)

    def add_checkpoint(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._checkpoints.append(data)
        self._broadcast("checkpoint", data)

    # -- SSE subscription --

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, event: str, data: Any) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                pass
