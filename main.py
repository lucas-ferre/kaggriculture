"""Kaggle entry point. No network, credentials or external services at runtime."""
import sys
from pathlib import Path

# Supports both local file-based loading and /kaggle_simulations/agent/main.py.
_root = str(Path(__file__).resolve().parent) if "__file__" in globals() else "/kaggle_simulations/agent"
if _root not in sys.path:
    sys.path.insert(0, _root)

from kaggriculture_agent.agent import FarmAgent

_players = {}


def agent(observation, configuration=None):
    player = int(observation["player"])
    if player not in _players:
        _players[player] = FarmAgent()
    return _players[player](observation, configuration)
