"""Opponent Elixir & Card Cycle Tracker for py-clash-bot.

Adapted from NezarAli/ElixirCounter and ClashRoyaleAi state estimation:
- Tracks opponent's elixir regeneration rate based on game phase (1x, 2x, 3x elixir).
- Deducts elixir when new enemy units/cards are detected on the arena.
- Tracks opponent's card cycle (4 cards in hand, 4 in queue; a played card cannot reappear
  for 4 card plays).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyclashbot.bot.board import Unit

from pyclashbot.bot.board import Tower
from pyclashbot.bot.card_knowledge import facts_for

# Elixir regeneration intervals in seconds per 1 elixir:
ELIXIR_REGEN_SINGLE = 2.8
ELIXIR_REGEN_DOUBLE = 1.4
ELIXIR_REGEN_TRIPLE = 0.9

MAX_ELIXIR = 10.0
STARTING_ELIXIR = 5.0
CYCLE_LOCK_COUNT = 4  # A played card needs 4 other cards played to return to hand


@dataclass
class OpponentTracker:
    """Maintains estimated opponent elixir and card rotation state."""

    current_elixir: float = STARTING_ELIXIR
    last_update_time: float = field(default_factory=time.time)
    match_start_time: float = field(default_factory=time.time)
    played_history: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    known_seen_units: dict[str, float] = field(default_factory=dict)
    enemy_tower_min_hp: dict[str, float] = field(
        default_factory=lambda: {"left": 1.0, "right": 1.0, "center": 1.0}
    )

    def reset(self) -> None:
        """Reset state at the start of a match."""
        now = time.time()
        self.current_elixir = STARTING_ELIXIR
        self.last_update_time = now
        self.match_start_time = now
        self.played_history.clear()
        self.known_seen_units.clear()
        self.enemy_tower_min_hp = {"left": 1.0, "right": 1.0, "center": 1.0}

    def update_enemy_towers(self, towers: Sequence[Tower]) -> tuple[Tower, ...]:
        """Track minimum HP seen so transient UI bar fadeouts do not reset damaged towers to 100%."""
        updated: list[Tower] = []
        for t in towers:
            min_hp = self.enemy_tower_min_hp.get(t.lane, 1.0)
            if t.readable and t.hp < min_hp:
                self.enemy_tower_min_hp[t.lane] = t.hp
                min_hp = t.hp
            effective_hp = min(t.hp, min_hp)
            is_readable = t.readable or min_hp < 1.0
            updated.append(t if (t.hp == effective_hp and t.readable == is_readable) else Tower(x=t.x, y=t.y, lane=t.lane, hp=effective_hp, readable=is_readable))
        return tuple(updated)

    def _regen_rate_for_elapsed(self, elapsed: float) -> float:
        """Elixir regenerated per second based on match clock."""
        if elapsed >= 240.0:
            return 1.0 / ELIXIR_REGEN_TRIPLE
        if elapsed >= 120.0:
            return 1.0 / ELIXIR_REGEN_DOUBLE
        return 1.0 / ELIXIR_REGEN_SINGLE

    def update(self, elapsed: float, units: Sequence[Unit]) -> None:
        """Update tracker with elapsed battle time and detected units."""
        now = time.time()
        dt = max(0.0, now - self.last_update_time)
        self.last_update_time = now

        # 1. Regenerate elixir
        rate = self._regen_rate_for_elapsed(elapsed)
        self.current_elixir = min(MAX_ELIXIR, self.current_elixir + rate * dt)

        # 2. Check for newly spawned enemy units
        for u in units:
            if not u.enemy or not u.card_id:
                continue
            card_name = (
                u.card_id.lower()
                .removeprefix("enemy_")
                .removeprefix("ally_")
                .strip()
                .replace("-", "_")
            )

            # Debounce: only count a unit once every 3.5 seconds per card type
            last_seen = self.known_seen_units.get(card_name, 0.0)
            if now - last_seen > 3.5:
                self.known_seen_units[card_name] = now
                self.on_opponent_play(card_name)

    def on_opponent_play(self, card_id: str) -> None:
        """Deduct elixir and record card in rotation when opponent plays a card."""
        facts = facts_for(card_id)
        cost = float(facts.elixir) if facts.elixir > 0 else 3.0

        self.current_elixir = max(0.0, self.current_elixir - cost)
        self.played_history.append(facts.card_id)

    def is_card_in_hand(self, card_id: str) -> bool:
        """Return False if the card was played within the last 4 plays (out of cycle)."""
        canonical_id = facts_for(card_id).card_id
        recent_four = list(self.played_history)[-CYCLE_LOCK_COUNT:]
        return canonical_id not in recent_four

    @property
    def opponent_win_condition(self) -> str | None:
        """Return the primary win condition identified from opponent's played cards."""
        for cid in reversed(self.played_history):
            facts = facts_for(cid)
            if facts.role == "wincon" or (facts.targets_buildings and not facts.is_spell):
                return facts.card_id
        return None

    def is_opponent_wincon_in_hand(self) -> bool:
        """True if the opponent's identified win condition is back in hand."""
        wincon = self.opponent_win_condition
        if wincon is None:
            return True  # Threat is assumed present if unknown
        return self.is_card_in_hand(wincon)

    def has_small_spell_in_hand(self) -> bool:
        """True if opponent has a small spell available in hand to counter Goblin Barrel."""
        small_spells = {"the_log", "log", "zap", "arrows", "barb_barrel", "snowball", "tornado", "void"}
        seen_small_spells = [c for c in self.played_history if c in small_spells]
        if not seen_small_spells:
            return True  # If never seen, assume they might have one
        return any(self.is_card_in_hand(spell) for spell in seen_small_spells)

    def recently_baited_small_spell(self) -> bool:
        """True if opponent played a small spell within their last 3 cards (cycle locked)."""
        small_spells = {"the_log", "log", "zap", "arrows", "barb_barrel", "snowball", "tornado", "void"}
        recent_three = list(self.played_history)[-3:]
        return any(c in small_spells for c in recent_three)

    def is_low_elixir(self) -> bool:
        """True if opponent has less than 3 elixir (vulnerable to counter-push)."""
        return self.current_elixir < 3.0

    def elixir_advantage(self, our_elixir: int) -> float:
        """Positive if we have more elixir than the opponent."""
        return float(our_elixir) - self.current_elixir
