"""Taunt King Log Bait Reinforcement Learning Trainer.

Simulates Clash Royale battles at ~140x real-time speed using clash-royale-simulator,
training a PPO policy for the exact 8-card Log Bait deck:
- Knight
- Princess
- IceSpirits
- GoblinGang
- InfernoTower
- Fireball
- GoblinBarrel
- Log
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

# Add simulator directory to Python path
SIM_DIR = Path(__file__).resolve().parent.parent / "external" / "clash-royale-simulator" / "src" / "clasher_new"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

# Change working directory temporarily to SIM_DIR so json files load
orig_cwd = os.getcwd()
os.chdir(str(SIM_DIR))

try:
    import environment  # ty: ignore[unresolved-import]
    from environment import CREnv  # ty: ignore[unresolved-import]
    from masked_spatial import LegalPlacement, MaskedSpatialPolicy  # ty: ignore[unresolved-import]
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from strategies import (  # ty: ignore[unresolved-import]
        DiverseOpponent,
    )
    from train_core import EpisodeReflection  # ty: ignore[unresolved-import]
finally:
    os.chdir(orig_cwd)

LOG_BAIT_DECK = [
    "Knight",
    "Princess",
    "IceSpirits",
    "GoblinGang",
    "InfernoTower",
    "Fireball",
    "GoblinBarrel",
    "Log",
]

OPPONENT_DECKS = [
    ["Giant", "Musketeer", "MiniPekka", "Minions", "Fireball", "Arrows", "Knight", "Archer"],
    ["Pekka", "Wizard", "Valkyrie", "Musketeer", "Minions", "Arrows", "Fireball", "Knight"],
    ["HogRider", "Fireball", "Arrows", "Minions", "Valkyrie", "Archer", "Knight", "Musketeer"],
]


class LogBaitEnv(LegalPlacement):
    """Clash Royale Environment locked to the user's Taunt King Log Bait deck."""

    def __init__(self, opponent_model=None, visualize=False):
        # Configure player 0 to strictly use Log Bait deck
        environment.player_0_deck = LOG_BAIT_DECK[:]
        environment.player_1_deck = random.choice(OPPONENT_DECKS)[:]
        base_env = EpisodeReflection(CREnv(opponent_model=opponent_model, visualize=visualize))
        super().__init__(base_env)

    def reset(self, *, seed=None, options=None):
        environment.player_0_deck = LOG_BAIT_DECK[:]
        environment.player_1_deck = random.choice(OPPONENT_DECKS)[:]
        return super().reset(seed=seed, options=options)


class WinRateCallback(BaseCallback):
    """Monitors battle win rate, logs progress, and auto-saves checkpoints."""

    def __init__(
        self,
        check_freq: int = 1000,
        save_path: Path | None = None,
        verbose: int = 1,
        progress_callback: Callable[[dict], None] | None = None,
        stop_event: threading.Event | None = None,
        total_steps: int = 50_000,
    ) -> None:
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.total_steps = total_steps
        self.episode_count = 0
        self.wins = 0

    def _on_step(self) -> bool:
        if self.stop_event and self.stop_event.is_set():
            if self.progress_callback:
                self.progress_callback({"type": "stopped", "text": "[RL] Simulation interrupted by user."})
            return False

        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            self.episode_count += 1
            # Check winner from environment battle state
            envs = getattr(self.training_env, "envs", None)
            env = envs[0].unwrapped if envs else None
            if env is not None and hasattr(env, "battle") and env.battle.winner == 0:
                self.wins += 1

        if self.n_calls % self.check_freq == 0:
            win_rate = (self.wins / self.episode_count * 100.0) if self.episode_count > 0 else 0.0
            msg = (
                f"[Step {self.n_calls}/{self.total_steps}] Episodes: {self.episode_count} | "
                f"Log Bait Win Rate: {win_rate:.1f}% ({self.wins}/{self.episode_count})"
            )
            print(msg, flush=True)
            if self.progress_callback:
                self.progress_callback({
                    "type": "progress",
                    "step": self.n_calls,
                    "total_steps": self.total_steps,
                    "episodes": self.episode_count,
                    "wins": self.wins,
                    "win_rate": win_rate,
                    "text": msg,
                })
            if self.save_path:
                try:
                    self.model.save(str(self.save_path))
                    save_msg = f"[Auto-Save] Checkpoint saved to {self.save_path.name} at step {self.n_calls}"
                    print(save_msg, flush=True)
                    if self.progress_callback:
                        self.progress_callback({"type": "save", "text": save_msg})
                except Exception:
                    pass
        return True


def train(
    total_timesteps: int = 50_000,
    model_save_path: str = "models/log_bait_ppo.zip",
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    opponent_type: str = "counterpush",
) -> None:
    """Run continuous PPO simulation training with automatic checkpoints."""
    start_msg = f"[RL] Starting Log Bait RL Training for {total_timesteps} steps (~140x fast-forward)...\nDeck: {', '.join(LOG_BAIT_DECK)}"
    print(start_msg, flush=True)
    if progress_callback:
        progress_callback({"type": "start", "text": start_msg, "total_steps": total_timesteps})

    out_path = Path(orig_cwd) / model_save_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(str(SIM_DIR))
    try:
        # Train against diverse counter-push / defensive strategy
        opponent = DiverseOpponent(opponent_type)
        env = LogBaitEnv(opponent_model=opponent)

        if out_path.exists():
            resume_msg = f"[RL] Resuming from existing checkpoint: {out_path.name}..."
            print(resume_msg, flush=True)
            if progress_callback:
                progress_callback({"type": "info", "text": resume_msg})
            model = PPO.load(
                str(out_path),
                env=env,
                device="auto",
                learning_rate=3e-4,
            )
        else:
            init_msg = "[RL] Initializing fresh MaskedSpatialPolicy for Taunt King Log Bait..."
            print(init_msg, flush=True)
            if progress_callback:
                progress_callback({"type": "info", "text": init_msg})
            model = PPO(
                MaskedSpatialPolicy,
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                device="auto",
            )

        callback = WinRateCallback(
            check_freq=1000,
            save_path=out_path,
            progress_callback=progress_callback,
            stop_event=stop_event,
            total_steps=total_timesteps,
        )
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        elapsed = max(0.1, time.time() - start_time)

        # Final save
        model.save(str(out_path))
        done_msg = f"[RL] Training completed in {elapsed:.1f}s ({total_timesteps/elapsed:.1f} steps/s)!\nSaved policy checkpoint to: {out_path}"
        print(done_msg, flush=True)
        if progress_callback:
            progress_callback({
                "type": "finished",
                "text": done_msg,
                "elapsed": elapsed,
                "speed": total_timesteps / elapsed,
                "checkpoint": str(out_path),
            })
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Log Bait RL Policy in Simulation")
    parser.add_argument("--steps", type=int, default=50_000, help="Total simulation timesteps")
    parser.add_argument("--save", type=str, default="models/log_bait_ppo.zip", help="Path to save model")
    args = parser.parse_args()

    train(total_timesteps=args.steps, model_save_path=args.save)
