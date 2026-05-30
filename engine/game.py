"""Compatibility shim — PylosGame now lives in games.pylos.game."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from games.pylos.game import PylosGame, LEVEL_SIZES  # noqa: F401
