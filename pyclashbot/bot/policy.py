from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import torch
from torch import nn

from pyclashbot.bot.card_detection import PLAY_COORDS

SCREEN_WIDTH = 419
SCREEN_HEIGHT = 633
USE_CONTINUOUS_COORDS = os.getenv("PYCLASHBOT_CONTINUOUS_COORDS", "1") not in {"0", "false", "False"}
CONTINUOUS_SAMPLES = int(os.getenv("PYCLASHBOT_CONTINUOUS_SAMPLES", "18"))
REWARD_SCALE = 1.5
STATE_EXTRA_DIM = 19
STATE_BASE_DIM = 3 + 4 + STATE_EXTRA_DIM  # 26: elapsed, elixir, side + 4-card mask + 19 extras
ACTION_DIM = 4                            # card_idx, group_id, x_norm, y_norm
SAFE_BOUNDS_DEFAULT = (60, 360, 170, 480)
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "models" / "policy.pt"

SPELL_GROUPS = {
    "spell",
    "earthquake",
    "fireball",
    "freeze",
    "poison",
    "arrows",
    "snowball",
    "zap",
    "rocket",
    "lightning",
    "log",
    "tornado",
    "goblin_drill",
    "graveyard",
    "ground_spell",
    "lane_spell",
    "large_spell",
    "reactive_spell",
}
FORWARD_ALLOWED_GROUPS = SPELL_GROUPS


@dataclass(frozen=True)
class PolicyAction:
    card_index: int
    card_group: str
    coord: tuple[int, int]
    card_id: str | None = None


def _group_index_map() -> dict[str, int]:
    groups = sorted(PLAY_COORDS.keys())
    groups.append("No group")
    return {name: idx for idx, name in enumerate(groups)}


GROUP_INDEX = _group_index_map()


def candidate_coords_for_group(card_group: str, side_preference: str, elapsed_time: float) -> list[tuple[int, int]]:
    def _sample_uniform(x_min: int, x_max: int, y_min: int, y_max: int, count: int) -> list[tuple[int, int]]:
        if x_max <= x_min or y_max <= y_min:
            return []
        return [(int(random.uniform(x_min, x_max)), int(random.uniform(y_min, y_max))) for _ in range(max(1, count))]

    def _fallback_grid() -> list[tuple[int, int]]:
        if elapsed_time < 15:
            y_min, y_max = 440, 470
        elif elapsed_time < 80:
            y_min, y_max = 350, 450
        else:
            y_min, y_max = 290, 450

        if side_preference == "left":
            x_min, x_max = 60, 206
        else:
            x_min, x_max = 210, 351
        return _sample_uniform(x_min, x_max, y_min, y_max, CONTINUOUS_SAMPLES)

    if card_group != "No group":
        group_datum = PLAY_COORDS.get(card_group, {})
        if side_preference in group_datum:
            base = list(group_datum[side_preference])
            if card_group not in SPELL_GROUPS:
                if card_group in ("turret", "defense_building", "inferno_tower"):
                    x_min, x_max = 180, 240
                    y_min, y_max = 320, 380
                else:
                    x_min, x_max = (60, 206) if side_preference == "left" else (210, 351)
                    y_min, y_max = (360, 460) if elapsed_time < 80 else (300, 460)
                base.extend(_sample_uniform(x_min, x_max, y_min, y_max, CONTINUOUS_SAMPLES))
            return list(dict.fromkeys(base))
        if "coords" in group_datum:
            base = list(group_datum["coords"])
            return list(dict.fromkeys(base))

    return _fallback_grid()


def candidate_coords_free(bounds: tuple[int, int, int, int] | None = None) -> list[tuple[int, int]]:
    x_min, x_max, y_min, y_max = bounds or SAFE_BOUNDS_DEFAULT
    margin = 35
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin
    if x_max <= x_min or y_max <= y_min:
        x_min, x_max, y_min, y_max = SAFE_BOUNDS_DEFAULT
    return [(int(random.uniform(x_min, x_max)), int(random.uniform(y_min, y_max))) for _ in range(max(1, CONTINUOUS_SAMPLES))]


def safe_bounds_for_image(image) -> tuple[int, int, int, int]:
    if image is None or getattr(image, "shape", None) is None:
        return SAFE_BOUNDS_DEFAULT
    h, w = image.shape[:2]
    x_min = int(w * (60 / 419))
    x_max = int(w * (360 / 419))
    y_min = int(h * (120 / 633))
    y_max = int(h * (520 / 633))
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))
    if x_max <= x_min or y_max <= y_min:
        return SAFE_BOUNDS_DEFAULT
    return (x_min, x_max, y_min, y_max)


def build_state_vector(
    elapsed_time: float,
    elixir: int,
    side_preference: str,
    available_mask: Iterable[int],
    enemy_activity: float = 0.0,
    wincon_available: float = 0.0,
    enemy_red_top: float = 0.0,
    enemy_red_mid: float = 0.0,
    enemy_template_top: float = 0.0,
    enemy_template_mid: float = 0.0,
    lane_bias: float = 0.5,
    defend_mode: float = 0.0,
    player_hp: float = 1.0,
    enemy_hp: float = 1.0,
    player_hp_delta: float = 0.0,
    enemy_hp_delta: float = 0.0,
    player_left_delta: float = 0.0,
    player_right_delta: float = 0.0,
    player_king_delta: float = 0.0,
    enemy_left_delta: float = 0.0,
    enemy_right_delta: float = 0.0,
    enemy_king_delta: float = 0.0,
    crown_diff: float = 0.0,
) -> list[float]:
    side_flag = 1.0 if side_preference == "right" else 0.0
    base = [elapsed_time / 300.0, elixir / 10.0, side_flag]
    mask = [float(v) for v in available_mask]
    extras = [
        float(max(0.0, min(1.0, enemy_activity))),
        float(max(0.0, min(1.0, wincon_available))),
        float(max(0.0, min(1.0, enemy_red_top))),
        float(max(0.0, min(1.0, enemy_red_mid))),
        float(max(0.0, min(1.0, enemy_template_top))),
        float(max(0.0, min(1.0, enemy_template_mid))),
        float(max(0.0, min(1.0, lane_bias))),
        float(max(0.0, min(1.0, defend_mode))),
        float(max(0.0, min(1.0, player_hp))),
        float(max(0.0, min(1.0, enemy_hp))),
        float(max(-1.0, min(1.0, player_hp_delta))),
        float(max(-1.0, min(1.0, enemy_hp_delta))),
        float(max(-1.0, min(1.0, player_left_delta))),
        float(max(-1.0, min(1.0, player_right_delta))),
        float(max(-1.0, min(1.0, player_king_delta))),
        float(max(-1.0, min(1.0, enemy_left_delta))),
        float(max(-1.0, min(1.0, enemy_right_delta))),
        float(max(-1.0, min(1.0, enemy_king_delta))),
        float(max(-1.0, min(1.0, crown_diff))),
    ]
    return base + mask + extras


def build_action_vector(action: PolicyAction) -> list[float]:
    group_id = GROUP_INDEX.get(action.card_group, GROUP_INDEX["No group"])
    x, y = action.coord
    return [float(action.card_index), float(group_id), x / SCREEN_WIDTH, y / SCREEN_HEIGHT]


def hold_action() -> PolicyAction:
    return PolicyAction(card_index=-1, card_group="No group", coord=(0, 0))


def build_input_batch(state_vec: list[float], actions: list[PolicyAction]) -> list[list[float]]:
    return [state_vec + build_action_vector(action) for action in actions]


class EpsilonGreedyPolicy:
    def __init__(self, model, epsilon: float = 0.1):
        self.model = model
        self.epsilon = float(max(0.0, min(1.0, epsilon)))

    def select_action(self, state_vec: list[float], actions: list[PolicyAction]) -> PolicyAction | None:
        if not actions:
            return None
        if self.model is None or random.random() < self.epsilon:
            return random.choice(actions)

        if torch is None:
            return random.choice(actions)

        batch = build_input_batch(state_vec, actions)
        inputs = torch.tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(inputs)
        if outputs is None:
            return random.choice(actions)

        if outputs.dim() == 2 and outputs.shape[1] > 1:
            scores = outputs.max(dim=1).values
        else:
            scores = outputs.view(-1)
        best_idx = int(torch.argmax(scores).item())
        return actions[best_idx]


if nn is not None:

    class SimplePolicyNet(nn.Module):
        def __init__(self, input_dim: int = STATE_BASE_DIM + ACTION_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

else:

    class SimplePolicyNet:
        def __init__(self, input_dim: int = STATE_BASE_DIM + ACTION_DIM):
            raise RuntimeError("PyTorch is not installed.")


def load_torch_model(
    path: str | Path | None = None,
    logger=None,
    trainable: bool = False,
    input_dim: int = STATE_BASE_DIM + ACTION_DIM,
):
    if torch is None:
        return None
    target_path = Path(path) if path else DEFAULT_POLICY_PATH
    if not target_path.exists():
        return SimplePolicyNet(input_dim) if trainable else None

    try:
        state = torch.load(target_path, map_location="cpu")
        model = SimplePolicyNet(input_dim)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        if logger:
            logger.log(f"[Policy] Failed to load model from {target_path}: {exc}")
        return SimplePolicyNet(input_dim) if trainable else None


def save_torch_model(model, path: str | Path | None = None, logger=None):
    if torch is None or model is None:
        return
    target_path = Path(path) if path else DEFAULT_POLICY_PATH
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target_path)
    except Exception as exc:
        if logger:
            logger.log(f"[Policy] Failed to save model: {exc}")


class OnlineTrainer:
    def __init__(self, model, lr: float = 5e-4, model_path: str | Path | None = None, logger=None):
        self.model = model
        self.model_path = Path(model_path) if model_path else DEFAULT_POLICY_PATH
        self.logger = logger
        self.replay_buffer: list[tuple[list[float], PolicyAction, float]] = []
        self.replay_max = 128
        self.buffer: list[list[float]] = []
        self.action_count = 0
        self.spell_count = 0
        self.elixir_sum = 0.0
        self.elixir_samples = 0
        self.used_indices: set[int] = set()
        self.repeat_streak = 0
        self.last_card_index: int | None = None
        self.hold_count = 0
        self.step_updates = 0
        self._last_autosave = 0
        self.enemy_damage = 0.0
        self.player_damage = 0.0
        self.defensive_plays = 0
        self.counter_push_chain = 0
        self._last_defensive_elapsed: float | None = None
        self._last_defensive_side: str | None = None

        if torch is not None and model is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.optimizer = None
            self.loss_fn = None

    def update_step(self, state_vec: list[float], action: PolicyAction, reward: float) -> None:
        """Immediate per-play update with a mini-batch from experience replay."""
        if self.model is None or torch is None or self.optimizer is None or self.loss_fn is None:
            return
        if not state_vec:
            return

        self.replay_buffer.append((state_vec, action, float(reward)))
        if len(self.replay_buffer) > self.replay_max:
            self.replay_buffer = self.replay_buffer[-self.replay_max:]

        batch = [self.replay_buffer[-1]]
        if len(self.replay_buffer) >= 8:
            batch = random.sample(self.replay_buffer, k=8)

        self.model.train()
        inputs = torch.tensor(
            [s + build_action_vector(a) for (s, a, _r) in batch],
            dtype=torch.float32,
        )
        targets = torch.tensor([[float(r)] for (_s, _a, r) in batch], dtype=torch.float32)
        out = self.model(inputs)
        if out.dim() == 1:
            out = out.view(-1, 1)
        loss = self.loss_fn(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        self.step_updates += 1

        if self.logger and hasattr(self.logger, "add_policy_step"):
            try:
                self.logger.add_policy_step(reward)
            except Exception:
                pass

        if self.step_updates - self._last_autosave >= 25:
            save_torch_model(self.model, self.model_path, logger=self.logger)
            self._last_autosave = self.step_updates

    def record(
        self,
        state_vec: list[float],
        action: PolicyAction,
        elapsed_time: float | None = None,
        enemy_hp: float | None = None,
        player_hp: float | None = None,
        enemy_activity: float | None = None,
    ) -> None:
        if self.model is None:
            return
        self.buffer.append(state_vec + build_action_vector(action))
        if action.card_index < 0:
            self.hold_count += 1
        else:
            self.action_count += 1
            self.used_indices.add(action.card_index)
            if self.last_card_index == action.card_index:
                self.repeat_streak += 1
            else:
                self.repeat_streak = 0
            self.last_card_index = action.card_index

        if action.card_group in SPELL_GROUPS:
            self.spell_count += 1

        if len(state_vec) > 1:
            self.elixir_sum += float(state_vec[1]) * 10.0
            self.elixir_samples += 1

        if enemy_activity and enemy_activity > 0.05 and action.coord[1] >= 340:
            self.defensive_plays += 1

        if action.card_index >= 0 and elapsed_time is not None:
            side = "left" if action.coord[0] <= 210 else "right"
            if (
                self._last_defensive_elapsed is not None
                and (elapsed_time - self._last_defensive_elapsed) <= 8.0
                and side == self._last_defensive_side
                and action.card_group not in SPELL_GROUPS
            ):
                self.counter_push_chain += 1

            if action.coord[1] >= 360 and action.card_group not in SPELL_GROUPS:
                self._last_defensive_elapsed = elapsed_time
                self._last_defensive_side = side

    def _compute_shaped_reward(self, base: float, crowns: tuple[int, int] | None) -> float:
        reward = base
        if crowns:
            player, opp = crowns
            reward += 0.35 * (player - opp)
            if player >= 3:
                reward += 0.6
            if opp >= 3:
                reward -= 0.6

        reward -= 0.01 * self.action_count
        reward -= 0.03 * self.spell_count

        unique_cards = len(self.used_indices)
        reward += 0.03 * unique_cards
        reward -= 0.02 * max(0, self.repeat_streak)
        reward += 0.15 * self.defensive_plays
        reward += 0.08 * self.counter_push_chain
        return reward

    def update(self, reward: float, crowns: tuple[int, int] | None = None) -> None:
        if not self.buffer or self.model is None or torch is None or self.optimizer is None or self.loss_fn is None:
            self.buffer.clear()
            self._reset_stats()
            return

        shaped = self._compute_shaped_reward(reward, crowns) * REWARD_SCALE
        self.model.train()
        inputs = torch.tensor(self.buffer, dtype=torch.float32)
        targets = torch.full((inputs.shape[0], 1), float(shaped), dtype=torch.float32)
        outputs = self.model(inputs)
        if outputs.dim() == 1:
            outputs = outputs.view(-1, 1)
        loss = self.loss_fn(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        if self.logger:
            self.logger.change_status(
                f"[Policy] Online update: Reward={shaped:.3f} (base={reward:.1f}), Loss={loss.item():.4f}",
            )
            if hasattr(self.logger, "add_policy_progress"):
                try:
                    self.logger.add_policy_progress(shaped, loss.item())
                except Exception:
                    pass

        save_torch_model(self.model, self.model_path, logger=self.logger)
        self.buffer.clear()
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.action_count = 0
        self.spell_count = 0
        self.elixir_sum = 0.0
        self.elixir_samples = 0
        self.used_indices.clear()
        self.repeat_streak = 0
        self.last_card_index = None
        self.hold_count = 0
        self.defensive_plays = 0
        self.counter_push_chain = 0
        self._last_defensive_elapsed = None
        self._last_defensive_side = None
