"""Bridge between live py-clash-bot BoardState and the trained Log Bait PPO model."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyclashbot.bot.decide import PlayIntent

if TYPE_CHECKING:
    from pyclashbot.bot.board import BoardState

BRIDGE_Y = 283

SIM_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "clash-royale-simulator" / "src" / "clasher_new"
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "log_bait_ppo.zip"

_MODEL = None
_ENV_MODULE = None


def _load_model():
    global _MODEL, _ENV_MODULE
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        return None

    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))

    orig_cwd = os.getcwd()
    try:
        os.chdir(str(SIM_DIR))
        import environment  # ty: ignore[unresolved-import]
        from stable_baselines3 import PPO
        _ENV_MODULE = environment
        _MODEL = PPO.load(str(MODEL_PATH), device="cpu")
    except Exception as e:
        print(f"Warning: Failed to load Log Bait RL model: {e}")
        _MODEL = None
    finally:
        os.chdir(orig_cwd)

    return _MODEL


CARD_NAME_TO_SIM = {
    "goblin_barrel": "GoblinBarrel",
    "knight": "Knight",
    "hero_knight": "Knight",
    "princess": "Princess",
    "goblin_gang": "GoblinGang",
    "inferno_tower": "InfernoTower",
    "fireball": "Fireball",
    "ice_spirit": "IceSpirits",
    "log": "Log",
    "the_log": "Log",
}

SIM_TO_CARD_NAME = {v: k for k, v in CARD_NAME_TO_SIM.items()}


class LogBaitMLBridge:
    """Predicts next play using the trained simulation RL model."""

    def __init__(self):
        self.model = _load_model()
        self.history_grid = []

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, board: BoardState, hand: list[str]) -> PlayIntent | None:
        if not self.is_available() or not hand:
            return None

        # Build 32x18x15 grid from live BoardState
        grid = np.zeros((32, 18, 15), dtype=np.float32)

        # Populate friendly and enemy towers
        for tower in board.own_towers:
            if not tower.alive:
                continue
            x_g = int(np.clip((tower.x / 419.0) * 18, 0, 17))
            y_g = int(np.clip((tower.y / 633.0) * 32, 0, 31))
            eid = 10 if tower.lane == "center" else 9
            grid[y_g, x_g, 0] = eid
            grid[y_g, x_g, 2] = 0
            grid[y_g, x_g, 9] = max(0.1, tower.hp)

        for tower in board.enemy_towers:
            if not tower.alive:
                continue
            x_g = int(np.clip((tower.x / 419.0) * 18, 0, 17))
            y_g = int(np.clip((tower.y / 633.0) * 32, 0, 31))
            eid = 10 if tower.lane == "center" else 9
            grid[y_g, x_g, 0] = eid
            grid[y_g, x_g, 2] = 1
            grid[y_g, x_g, 9] = max(0.1, tower.hp)

        # Populate detected units
        for unit in board.units:
            x_g = int(np.clip((unit.x / 419.0) * 18, 0, 17))
            y_g = int(np.clip((unit.y / 633.0) * 32, 0, 31))
            owner = 1 if unit.enemy else 0
            grid[y_g, x_g, 2] = owner
            grid[y_g, x_g, 5] = 1.0 if unit.is_air else 0.0

        if not self.history_grid:
            self.history_grid = [grid.copy() for _ in range(8)]
        else:
            self.history_grid.append(grid)
            if len(self.history_grid) > 8:
                self.history_grid.pop(0)

        # Convert hand to entity indices
        entity_names = _ENV_MODULE.entity_names if _ENV_MODULE else []
        hand_indices = []
        for c in hand[:4]:
            sim_name = CARD_NAME_TO_SIM.get(c, "None")
            idx = entity_names.index(sim_name) if sim_name in entity_names else 0
            hand_indices.append(idx)
        while len(hand_indices) < 5:
            hand_indices.append(0)

        # Build legal mask: available cards can be played
        legal_mask = np.ones((4, 32, 18), dtype=np.int8)

        obs = {
            "grid": np.stack(self.history_grid).astype(np.float32),
            "hand": np.array(hand_indices, dtype=np.int32),
            "elixir": np.array([float(board.elixir)], dtype=np.float32),
            "phase": 0,
            "time_till_next_phase": np.array([0.5], dtype=np.float32),
            "legal_mask": legal_mask,
        }

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            action_arr = np.asarray(action).reshape(-1)
            slot = int(action_arr[0])
            y_g = int(action_arr[1])
            x_g = int(action_arr[2])

            if slot == 0 or slot > len(hand):
                return None  # Wait / no-op

            chosen_card = hand[slot - 1]

            # Convert 18x32 grid back to screen pixels (419x633)
            # If target is Goblin Barrel, drop right on the lowest HP princess tower
            if chosen_card == "goblin_barrel" and board.weakest_enemy_tower:
                px = board.weakest_enemy_tower.x
                py = board.weakest_enemy_tower.y
            else:
                px = int((x_g / 18.0) * 419)
                py = int((y_g / 32.0) * 633)
                # Keep friendly troops on friendly side of the river if not spell
                if chosen_card not in ("goblin_barrel", "fireball", "log", "the_log"):
                    py = max(BRIDGE_Y, py)

            return PlayIntent(
                card_id=chosen_card,
                x=px,
                y=py,
                score=500.0,
                rule="LogBait_RL_PPO",
                reason=f"PPO simulation policy choice (slot={slot}, grid=({x_g},{y_g}))",
                tags=("simulation", "ppo", "rl"),
            )
        except Exception as e:
            print(f"LogBait ML prediction error: {e}")
            return None
