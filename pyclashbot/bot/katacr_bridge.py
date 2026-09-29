import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from pyclashbot.bot.board import BoardState
from pyclashbot.bot.decide import PlayIntent

# Configure JAX to use CPU and not preallocate all memory
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

kata_path = Path(__file__).parents[2] / "KataCR"
if str(kata_path) not in sys.path:
    sys.path.append(str(kata_path))

try:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    from katacr.constants.label_list import unit_list  # ty: ignore[unresolved-import]
    from katacr.policy.offline.starformer import StARConfig, StARformer  # ty: ignore[unresolved-import]
    _JAX_AVAILABLE = True
except ImportError as e:
    _JAX_AVAILABLE = False
    print(f"KataCR JAX imports failed: {e}")


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


class KataCRBridge:
    """Singleton bridge to interface py-clash-bot state with StARformer model."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        if not _JAX_AVAILABLE:
            self.model = None
            return

        print("Initializing StARformer KataCR Bridge...")
        path_weights = kata_path / "runs" / "StARformer_3L__step50" / "ckpt"

        ckpt_mngr = ocp.CheckpointManager(
            str(path_weights),
            item_names=('variables', 'config'),
            options=ocp.CheckpointManagerOptions(step_format_fixed_length=3)
        )

        epochs = [int(p.name) for p in path_weights.glob('*') if p.is_dir() and p.name.isdigit()]
        if not epochs:
            print("No StARformer checkpoints found.")
            self.model = None
            return

        latest_epoch = max(epochs)

        # Load config
        load_info = ckpt_mngr.restore(latest_epoch, args=ocp.args.Composite(
            config=ocp.args.JsonRestore()
        ))
        self.cfg = load_info['config']

        if 'cnn_mode' not in self.cfg:
            self.cfg['cnn_mode'] = 'resnet'

        self.model = StARformer(StARConfig(**self.cfg))
        self.rng = jax.random.PRNGKey(42)

        # Initialize dummy variables to get target tree structure for loading
        b_size, seq_len = 1, self.cfg.get('n_step', 50)
        s = {
            'arena': jnp.empty((b_size, seq_len, 32, 18, 1+1+self.cfg.get('n_bar_size', 192)*2), int),
            'arena_mask': jnp.empty((b_size, seq_len, 32, 18), bool),
            'cards': jnp.empty((b_size, seq_len, 5), int),
            'elixir': jnp.empty((b_size, seq_len), int)
        }
        a = {'select': jnp.empty((b_size, seq_len), int), 'pos': jnp.empty((b_size, seq_len, 2), int)}
        r = jnp.empty((b_size, seq_len), float)
        timestep = jnp.empty((b_size, seq_len), int)

        variables = self.model.init(self.rng, s, a, r, timestep, False)

        # Load weights
        load_info = ckpt_mngr.restore(latest_epoch, args=ocp.args.Composite(
            variables=ocp.args.StandardRestore(variables),
            config=ocp.args.JsonRestore()
        ))
        params = load_info['variables']['params']

        self.model.create_fns()
        from katacr.policy.offline.starformer import TrainConfig  # ty: ignore[unresolved-import]
        dummy_cfg = TrainConfig(steps_per_epoch=1, n_step=self.cfg.get('n_step', 50), accumulate=1)
        self.state = self.model.get_state(dummy_cfg, train=False)
        self.state = self.state.replace(params=params)

        self.n_step = self.cfg.get('n_step', 50)

        # History buffers for autoregressive context
        self.history_s_arena = deque(maxlen=self.n_step)
        self.history_s_arena_mask = deque(maxlen=self.n_step)
        self.history_s_cards = deque(maxlen=self.n_step)
        self.history_s_elixir = deque(maxlen=self.n_step)

        self.history_a_select = deque(maxlen=self.n_step)
        self.history_a_pos = deque(maxlen=self.n_step)

        self.history_r = deque(maxlen=self.n_step)
        self.history_timestep = deque(maxlen=self.n_step)

        self.start_time = time.time()
        print("StARformer loaded successfully.")

    def _map_card_name(self, card_id: str) -> int:
        """Map py-clash-bot string card ID to KataCR unit_list index."""
        name = card_id.lower().replace("_", "-")
        # Exceptions
        exceptions = {
            "arrow": "arrows",
            "battleram": "battle-ram",
            "bombtower": "bomb-tower",
            "brawler": "goblin-brawler",
            "cagegoblin": "goblin-cage",
            "knight-evo": "knight-evolution",
            "log": "the-log",
            "royale-giant": "royal-giant",
            "wallbreaker": "wall-breaker"
        }
        name = exceptions.get(name, name)

        if name in unit_list:
            return unit_list.index(name)
        return 0  # 0 is EMPTY_CARD_INDEX

    def _convert_coords(self, px, py):
        """Convert py-clash-bot pixels (419x633) to KataCR grid (18x32)."""
        x_grid = int((px / 419.0) * 18)
        y_grid = int((py / 633.0) * 32)
        return min(max(x_grid, 0), 17), min(max(y_grid, 0), 31)

    def _convert_coords_back(self, x_grid, y_grid):
        """Convert KataCR grid (18x32) back to py-clash-bot pixels (419x633)."""
        px = int((x_grid / 18.0) * 419)
        py = int((y_grid / 32.0) * 633)
        return px, py

    def predict(self, board: BoardState, hand: list[str]) -> PlayIntent | None:
        if self.model is None:
            return None

        arena = np.zeros((32, 18, 1+1+self.cfg.get('n_bar_size', 192)*2), dtype=np.int32)
        arena_mask = np.zeros((32, 18), dtype=bool)

        # Populate arena with units
        for unit in board.units:
            x_g, y_g = self._convert_coords(unit.x, unit.y)
            bel = -1 if unit.enemy else 1
            cls_id = self._map_card_name(unit.card_id or "skeleton")

            arena[y_g, x_g, 0] = cls_id
            arena[y_g, x_g, 1] = bel
            arena_mask[y_g, x_g] = True

        # Hand cards (padded to 5)
        cards = np.zeros(5, dtype=np.int32)
        for i, card in enumerate(hand[:4]):
            cards[i+1] = self._map_card_name(card)

        elixir = board.elixir

        # Push to history
        self.history_s_arena.append(arena)
        self.history_s_arena_mask.append(arena_mask)
        self.history_s_cards.append(cards)
        self.history_s_elixir.append(elixir)

        # Push dummy previous action if history is missing one
        if len(self.history_a_select) < len(self.history_s_arena):
            self.history_a_select.append(0)
            self.history_a_pos.append(np.array([-1, 0]))
            self.history_r.append(0.0)
            self.history_timestep.append(int(time.time() - self.start_time))

        # Construct arrays
        s_dict = {
            'arena': np.expand_dims(np.stack(list(self.history_s_arena)), 0),
            'arena_mask': np.expand_dims(np.stack(list(self.history_s_arena_mask)), 0),
            'cards': np.expand_dims(np.stack(list(self.history_s_cards)), 0),
            'elixir': np.expand_dims(np.stack(list(self.history_s_elixir)), 0)
        }

        a_dict = {
            'select': np.expand_dims(np.stack(list(self.history_a_select)), 0),
            'pos': np.expand_dims(np.stack(list(self.history_a_pos)), 0)
        }

        rtg = np.expand_dims(np.zeros(len(self.history_r), dtype=np.float32), 0)
        timestep_arr = np.expand_dims(np.stack(list(self.history_timestep)), 0)

        # Pad along time axis
        def pad(x):
            return pad_along_axis(x, self.n_step, 1)

        s_padded = {k: pad(v) for k, v in s_dict.items()}
        a_padded = {k: pad(v) for k, v in a_dict.items()}
        rtg_padded = pad(rtg)
        ts_padded = pad(timestep_arr)

        step_len = len(self.history_s_arena)
        self.rng, rng_use = jax.random.split(self.rng)

        action, logits_select, logits_x, logits_y = jax.device_get(self.model.predict(
            self.state,
            s_padded,
            a_padded,
            rtg_padded,
            ts_padded,
            step_len, rng_use, True
        ))

        act = action[0]
        select_idx = int(act[0])

        # Logits -> probs
        prob_y = np.exp(logits_y-logits_y.max())[0].reshape(32, 1)
        prob_y /= prob_y.sum()
        prob_x = np.exp(logits_x-logits_x.max())[0].reshape(1, 18)
        prob_x /= prob_x.sum()
        prob_pos = prob_y * prob_x

        best_y, best_x = np.unravel_index(np.argmax(prob_pos), prob_pos.shape)
        px, py = self._convert_coords_back(best_x, best_y)

        # 0 is usually 'empty'/None. We want 1..4 corresponding to hand cards
        if select_idx == 0 or select_idx > len(hand):
            return None # Skip action

        chosen_card = hand[select_idx - 1]

        # Update our action history for autoregressive
        self.history_a_select[-1] = select_idx
        self.history_a_pos[-1] = np.array([best_y, best_x])

        return PlayIntent(
            card_id=chosen_card,
            x=px,
            y=py,
            score=999.0,
            rule="StARformer Offline RL",
            reason="Predicted by KataCR model",
            tags=("KataCR",)
        )
