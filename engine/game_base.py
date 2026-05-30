from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    """Abstract base class for board games compatible with AlphaZero training.

    Any game implementing this interface can be trained with MCTS self-play,
    evaluated with ELO ratings, and served via the WebSocket server.
    """

    action_space: int
    observation_shape: tuple

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, action, **kwargs):
        ...

    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        ...

    @abstractmethod
    def to_observation(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_result(self):
        ...

    @abstractmethod
    def get_first_person_result(self):
        ...

    @abstractmethod
    def swap_result(self, result):
        ...

    @abstractmethod
    def undo_last_action(self):
        ...

    @abstractmethod
    def has_move(self) -> bool:
        ...

    @property
    @abstractmethod
    def turn(self) -> int:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    # --- Server serialization (each game implements its own) ---

    @abstractmethod
    def board_to_json(self) -> dict:
        ...

    @abstractmethod
    def legal_moves_to_json(self) -> list[dict]:
        ...

    @abstractmethod
    def action_from_json(self, msg: dict) -> int | None:
        ...

    @abstractmethod
    def action_to_json(self, action: int) -> dict:
        ...

    @abstractmethod
    def make_state_msg(self) -> dict:
        ...

    def board_hash(self) -> bytes:
        return self.to_observation().tobytes()
