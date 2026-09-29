"""Pure board reading: one BGR frame in, a :class:`BoardState` out.

Everything here is a pure function of the frame. That is deliberate: the fight
recorder writes lossless frames to disk, so any reading can be replayed offline
against a recorded pack and asserted in a test without an emulator. Nothing in
this module touches an emulator, the clock, or global state.

What is readable today vs. what is not:

* elixir -- exact integer, from the fill length of the drawn bar.
* towers -- fixed positions (see ``coords``), plus a health-bar fill fraction.
* enemy units -- **not** detected. The field exists and is always empty, so the
  decision layer can be written and tested against it today; populating it needs
  a unit detector (a trained model), which is a separate piece of work.

Channel order is BGR, matching raw ``emulator.screenshot()`` and the rest of the
detection layer (``card_detection`` fingerprints rely on it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy

try:
    from pyclashbot.bot.vision import detect_enemy_units_yolo, is_yolo_available
except ImportError:
    # Handle the case before vision.py is fully created/available
    def detect_enemy_units_yolo(bgr): return ()
    def is_yolo_available(): return False

from pyclashbot.bot.coords import (
    ARENA_W,
    ELIXIR_BAR_MAX,
    ELIXIR_BAR_PIP_W,
    ELIXIR_BAR_ROW,
    ELIXIR_BAR_ZERO_EDGE,
    ENEMY_KING_TOWER,
    ENEMY_PRINCESS_TOWERS,
    ENEMY_TOWER_HP_BAR_BANDS,
    ENEMY_TOWER_HP_BAR_FULL_W,
    ENEMY_TOWER_HP_BAR_ROWS,
    LANE_X,
    OWN_KING_TOWER,
    OWN_PRINCESS_TOWERS,
    OWN_TOWER_HP_BAR_BANDS,
    OWN_TOWER_HP_BAR_FULL_W,
    OWN_TOWER_HP_BAR_ROWS,
    RIVER_Y,
)

# BGR. The elixir fill and the tower health bars are both this magenta family;
# a loose per-channel bound keeps the saturated UI art out without going brittle
# on the darker, partially-filled end of the bar.
_MAGENTA_MIN = (150, 60, 150)  # B, G, R
_HEALTHBAR_MIN_RED = 150
_HEALTHBAR_MAX_GREEN = 190


@dataclass(frozen=True)
class Tower:
    """A tower's fixed position and what we can read about its health."""

    x: int
    y: int
    lane: str  # "left" | "right" | "center"
    hp: float  # 1.0 = full, 0.0 = destroyed; 1.0 when unreadable
    readable: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


@dataclass(frozen=True)
class Unit:
    """An on-board troop. Unpopulated until a unit detector exists."""

    x: int
    y: int
    is_air: bool
    enemy: bool
    card_id: str | None = None


@dataclass(frozen=True)
class BoardState:
    """Everything the decision layer is allowed to look at.

    Deliberately narrow: no screenshots, no emulator, no clock. A decision is a
    pure function of this plus the hand, so it replays identically offline.
    """

    elixir: int
    elapsed: float
    double_elixir: bool
    own_towers: tuple[Tower, ...]
    enemy_towers: tuple[Tower, ...]
    units: tuple[Unit, ...] = field(default=())

    @property
    def enemy_princess(self) -> tuple[Tower, ...]:
        return tuple(t for t in self.enemy_towers if t.lane in ("left", "right"))

    @property
    def weakest_enemy_tower(self) -> Tower | None:
        """The enemy tower the bot should be pushing: lowest HP, ties to the left.

        Only considers living towers. Prefers living princess towers over king tower.
        """
        alive_princess = [t for t in self.enemy_princess if t.alive]
        if alive_princess:
            return min(alive_princess, key=lambda t: (t.hp, 0 if t.lane == "left" else 1))
        alive_towers = [t for t in self.enemy_towers if t.alive]
        if alive_towers:
            return min(alive_towers, key=lambda t: (t.hp, 0 if t.lane == "left" else 1))
        return None

    @property
    def own_princess(self) -> tuple[Tower, ...]:
        return tuple(t for t in self.own_towers if t.lane in ("left", "right"))

    @property
    def weakest_own_tower(self) -> Tower | None:
        """The friendly tower needing the most urgent defense."""
        alive_towers = [t for t in self.own_princess if t.alive]
        if not alive_towers:
            return None
        return min(alive_towers, key=lambda t: t.hp)

    @property
    def own_towers_alive(self) -> int:
        return sum(1 for t in self.own_towers if t.alive)

    @property
    def has_enemy_units(self) -> bool:
        return any(u.enemy for u in self.units)

    def enemy_units_in_half(self, enemy_half: bool) -> tuple[Unit, ...]:
        """Enemy units on the bot's side/bridge (y >= 195) or enemy backfield (y < 195)."""
        return tuple(
            u for u in self.units if u.enemy and ((u.y >= 195) if enemy_half else (u.y < 195))
        )


def _is_magenta(pixel: numpy.ndarray) -> bool:
    b, g, r = (int(v) for v in pixel[:3])
    return b >= _MAGENTA_MIN[0] and r >= _MAGENTA_MIN[2] and g <= 190


def read_elixir(bgr: numpy.ndarray) -> int:
    """Exact elixir count (0-10) from the drawn bar's fill length.

    The fill starts at the drop icon and grows rightward; its right edge is the
    only monotonic signal, so we measure that edge and convert with the pitch
    calibrated in ``coords``. Robust to card art overlapping the bar, which is
    what broke the single-pixel approach.
    """
    if bgr.shape[0] <= ELIXIR_BAR_ROW or bgr.shape[1] < ARENA_W:
        return 0
    row = bgr[ELIXIR_BAR_ROW]
    hits = numpy.nonzero([_is_magenta(row[x]) for x in range(ARENA_W)])[0]
    if hits.size == 0:
        return 0
    right_edge = float(hits.max())
    n = round((right_edge - ELIXIR_BAR_ZERO_EDGE) / ELIXIR_BAR_PIP_W)
    return int(max(0, min(ELIXIR_BAR_MAX, n)))


def _read_health_bar(bgr: numpy.ndarray, band: tuple[int, int], full_w: int) -> float | None:
    """Fill fraction of a horizontal health bar, or None when it is not drawn.

    The bar is magenta, and its length is the tower's remaining HP. Returns None
    (rather than a guess) when too few bar-coloured pixels are present, so a
    destroyed tower or a post-battle frame is never reported as healthy -- the
    caller maps None to ``readable=False``.
    """
    x0, x1 = band
    y0, y1 = ENEMY_TOWER_HP_BAR_ROWS
    if bgr.shape[0] < y1 or bgr.shape[1] < x1:
        return None
    strip = bgr[y0:y1, x0:x1]
    if strip.size == 0 or full_w <= 0:
        return None
    blue = strip[:, :, 0].astype(int)
    green = strip[:, :, 1].astype(int)
    red = strip[:, :, 2].astype(int)
    magenta = (red >= _HEALTHBAR_MIN_RED) & (blue >= _HEALTHBAR_MIN_RED) & (green <= _HEALTHBAR_MAX_GREEN)
    cols = numpy.nonzero(magenta.any(axis=0))[0]
    if cols.size < 3:
        return None
    return float(min(1.0, max(0.0, (cols.max() - cols.min() + 1) / full_w)))


def _read_friendly_health_bar(bgr: numpy.ndarray, band: tuple[int, int], full_w: int) -> float | None:
    """Fill fraction of an own (friendly) horizontal health bar, or None if not readable."""
    x0, x1 = band
    y0, y1 = OWN_TOWER_HP_BAR_ROWS
    if bgr.shape[0] < y1 or bgr.shape[1] < x1:
        return None
    strip = bgr[y0:y1, x0:x1]
    if strip.size == 0 or full_w <= 0:
        return None
    # Blue / cyan bar fill, plus white font digits inside the bar
    fill_mask = ((strip[:, :, 0] >= 180) & (strip[:, :, 0] > strip[:, :, 2] + 15)) | (
        (strip[:, :, 0] >= 200) & (strip[:, :, 1] >= 180) & (strip[:, :, 2] >= 150)
    )
    cols = numpy.nonzero(fill_mask.any(axis=0))[0]
    if cols.size < 2:
        return 0.0  # Tower destroyed (rubble)
    return float(min(1.0, max(0.0, (cols.max() - cols.min() + 1) / full_w)))


def read_towers(bgr: numpy.ndarray) -> tuple[tuple[Tower, ...], tuple[Tower, ...]]:
    """Read both tower sets.

    Friendly and enemy towers get real HP fractions measured from their health bars.
    """
    own: list[Tower] = []
    for lane, (x, y) in OWN_PRINCESS_TOWERS.items():
        frac = _read_friendly_health_bar(bgr, OWN_TOWER_HP_BAR_BANDS[lane], OWN_TOWER_HP_BAR_FULL_W[lane])
        own.append(Tower(x, y, lane, 1.0 if frac is None else frac, frac is not None))
    # Friendly King Tower
    x0_k, x1_k = OWN_TOWER_HP_BAR_BANDS["king"]
    strip_k = bgr[480:485, x0_k:x1_k] if bgr.shape[0] >= 485 and bgr.shape[1] >= x1_k else None
    if strip_k is not None and strip_k.size > 0:
        fill_k = ((strip_k[:, :, 0] >= 180) & (strip_k[:, :, 0] > strip_k[:, :, 2] + 15)) | (
            (strip_k[:, :, 0] >= 200) & (strip_k[:, :, 1] >= 180) & (strip_k[:, :, 2] >= 150)
        )
        cols_k = numpy.nonzero(fill_k.any(axis=0))[0]
        frac_k = (
            float(min(1.0, max(0.0, (cols_k.max() - cols_k.min() + 1) / OWN_TOWER_HP_BAR_FULL_W["king"])))
            if cols_k.size >= 2
            else 1.0
        )
    else:
        frac_k = 1.0
    own.append(Tower(OWN_KING_TOWER[0], OWN_KING_TOWER[1], "center", frac_k, True))

    enemy: list[Tower] = []
    for lane, (x, y) in ENEMY_PRINCESS_TOWERS.items():
        frac = _read_health_bar(bgr, ENEMY_TOWER_HP_BAR_BANDS[lane], ENEMY_TOWER_HP_BAR_FULL_W[lane])
        enemy.append(Tower(x, y, lane, 1.0 if frac is None else frac, frac is not None))
    frac = _read_health_bar(bgr, ENEMY_TOWER_HP_BAR_BANDS["king"], ENEMY_TOWER_HP_BAR_FULL_W["king"])
    enemy.append(
        Tower(ENEMY_KING_TOWER[0], ENEMY_KING_TOWER[1], "center", 1.0 if frac is None else frac, frac is not None)
    )
    return tuple(own), tuple(enemy)


def lane_x(lane: str) -> int:
    """Arena x for a lane name; ``center`` is the river/king column."""
    return LANE_X.get(lane, LANE_X["center"])


def is_double_elixir(elapsed: float) -> bool:
    """True from 1:00 (single elixir) and 2:00 (double/triple) onwards.

    RoyaleAPI carries no elixir-timing data, so these are the documented game
    constants, kept here as named values rather than inline magic numbers.
    """
    return elapsed >= 60.0


def detect_enemy_units(bgr: numpy.ndarray) -> tuple[Unit, ...]:
    """Detect enemy units and air threats on the arena from health bars and level tags.

    If YOLO vision is available and enabled, it will use deep learning to detect specific cards.
    Otherwise, it falls back to the fast red-pixel heuristic.
    """
    yolo_units = ()
    if is_yolo_available():
        yolo_units = detect_enemy_units_yolo(bgr)

    if bgr.shape[0] < 510 or bgr.shape[1] < 385:
        return yolo_units
    arena = bgr[145:510, 35:385]
    b = arena[:, :, 0].astype(int)
    g = arena[:, :, 1].astype(int)
    r = arena[:, :, 2].astype(int)

    red_mask = ((r >= 165) & (g <= 75) & (b <= 75)).astype(numpy.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask)

    units: list[Unit] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= 6 and (w >= 3 or h >= 3):
            cx = int(centroids[i][0]) + 35
            cy = int(centroids[i][1]) + 145

            # Deduplication: do not double-count units already detected by YOLO
            if any(abs(yu.x - cx) <= 30 and abs(yu.y - cy) <= 30 for yu in yolo_units):
                continue

            # Air check: unit is crossing river water outside bridges
            # (Left bridge is x: 85-135, right bridge is x: 285-335)
            is_air = False
            if 260 <= cy <= 315:
                if not ((85 <= cx <= 135) or (285 <= cx <= 335)):
                    is_air = True
            elif cy > RIVER_Y and area < 15 and w <= 8 and h <= 8:
                is_air = True

            units.append(Unit(x=cx, y=cy, is_air=is_air, enemy=True))
    return tuple(list(yolo_units) + units)


def read_board(bgr: numpy.ndarray, elapsed: float) -> BoardState:
    """Build a :class:`BoardState` from one BGR frame and the battle clock."""
    own, enemy = read_towers(bgr)
    return BoardState(
        elixir=read_elixir(bgr),
        elapsed=elapsed,
        double_elixir=is_double_elixir(elapsed),
        own_towers=own,
        enemy_towers=enemy,
        units=detect_enemy_units(bgr),
    )
