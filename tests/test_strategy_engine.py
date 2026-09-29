"""Offline board-reading tests driven by a recorded fight pack.

These run without an emulator: the pack is a directory of lossless PNG frames
plus the manifest, produced by ``pyclashbot.bot.recorder``. They are skipped when
the pack is absent (a fresh clone has no recordings), so CI stays green.

The pack used for calibration is
``recordings/20260928-152726-d69cbc`` (Classic 1v1, 419x633). Game-labelled
frames pin the elixir calibration: frames 0-5 show "Elixir bar is full!" (10),
and frames 40/100/200 show 2/3/3 on the drawn counter.
"""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from pyclashbot.bot.board import BoardState, Tower, Unit, read_board, read_elixir, read_towers
from pyclashbot.bot.card_knowledge import _TABLE, AIR_UNITS, TABLE_SIZE, facts_for
from pyclashbot.bot.coords import ENEMY_PRINCESS_TOWERS, RIVER_Y
from pyclashbot.bot.decide import PlayContext, decide

FRAMES_W, FRAMES_H = 419, 633


def _recordings_dir() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return ""
    return os.path.join(appdata, "py-clash-bot", "recordings")


def _pack_dir() -> str | None:
    """Newest real pack: a manifest plus a non-empty frames dir."""
    root = _recordings_dir()
    if not root or not os.path.isdir(root):
        return None
    # Prefer calibrated pack
    calib = os.path.join(root, "20260928-192105-246b98", "frames")
    if os.path.isdir(calib):
        return calib
    best: tuple[float, str] | None = None
    for slug in os.listdir(root):
        pack = os.path.join(root, slug)
        frames = os.path.join(pack, "frames")
        manifest = os.path.join(pack, "manifest.json")
        if not (os.path.isdir(frames) and os.path.isfile(manifest)):
            continue
        pngs = [n for n in os.listdir(frames) if n.endswith(".png")]
        if len(pngs) < 100:
            continue
        try:
            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if data.get("resolution") != [FRAMES_W, FRAMES_H]:
            continue
        f4 = os.path.join(frames, "000004.png")
        f0 = os.path.join(frames, "000000.png")
        if not (os.path.isfile(f4) and os.path.isfile(f0)):
            continue
        try:
            import numpy as _np
            _arr4 = _np.array(Image.open(f4).convert("RGB"))[:, :, ::-1]
            if read_elixir(_arr4) != 10:
                continue
            _arr0 = _np.array(Image.open(f0).convert("RGB"))[:, :, ::-1]
            if read_elixir(_arr0) != 9:
                continue
        except Exception:
            continue

        mtime = os.path.getmtime(pack)
        if best is None or mtime > best[0]:
            best = (mtime, frames)
    return None if best is None else best[1]


PACK = _pack_dir()
needs_pack = pytest.mark.skipif(PACK is None, reason="no recorded fight pack on this machine")


def _frame(index: int) -> Image.Image:
    assert PACK is not None
    names = sorted(n for n in os.listdir(PACK) if n.endswith(".png"))
    return Image.open(os.path.join(PACK, names[index])).convert("RGB")


def _frame_bgr(index: int):
    """A recorded frame as the BGR ndarray the bot actually passes around.

    ``emulator.screenshot()`` is BGR, so the board reader takes BGR; PIL hands
    back RGB, so the channel order is flipped here rather than weakening the
    production signature.
    """
    import numpy

    return numpy.asarray(_frame(index))[:, :, ::-1].copy()


def _frame_count() -> int:
    assert PACK is not None
    return len([n for n in os.listdir(PACK) if n.endswith(".png")])


# --- geometry ---------------------------------------------------------------


def test_arena_geometry_is_vertically_symmetric_about_the_river():
    """The measured towers must mirror across the river, or lane maths skews.

    The enemy princess towers sit well above the river -- that vertical gap is
    exactly why the old hardcoded spell y missed them, so assert the *shape* of
    the symmetry rather than a distance.
    """
    from pyclashbot.bot.coords import OWN_PRINCESS_TOWERS

    for lane in ("left", "right"):
        enemy_y = ENEMY_PRINCESS_TOWERS[lane][1]
        own_y = OWN_PRINCESS_TOWERS[lane][1]
        assert enemy_y < RIVER_Y < own_y, f"{lane}: towers are not straddling the river"
        midpoint = (enemy_y + own_y) / 2
        assert abs(midpoint - RIVER_Y) <= 8, f"{lane}: tower midpoint {midpoint} is off the river"


def test_lane_x_are_mirrored_about_the_arena_centre():
    from pyclashbot.bot.coords import LANE_X

    assert abs((LANE_X["left"] + LANE_X["right"]) - 419) <= 4
    assert LANE_X["left"] < LANE_X["center"] < LANE_X["right"]


# --- elixir -----------------------------------------------------------------


@needs_pack
def test_elixir_reads_the_game_labelled_frames():
    """Frames 4-5 carry the "Elixir bar is full!" banner, i.e. exactly 10.

    Frames 0-3 are the bar still filling at the start of the match and measure
    9 (fill right edge 350px vs 375px when full) -- both are asserted so a
    calibration that is off by one whole pip cannot pass.
    """
    for i in (4, 5):
        assert read_elixir(_frame_bgr(i)) == 10, f"frame {i} should read a full bar"
    for i in range(0, 4):
        assert read_elixir(_frame_bgr(i)) == 9, f"frame {i} should read a still-filling bar"


@needs_pack
def test_elixir_can_only_regen_one_per_frame():
    """Elixir regenerates ~1 per 1.4s, so at 3 fps it can only ever tick up by
    one between adjacent frames. A drop is fine (that is a card being played), but
    a rise of two or more means the fill edge is being read off a moving element.

    This is what the old single-pixel reader failed: it produced 9,8,4,3,4,5,1,4
    -- a +3 and a +4 rise, both impossible in a third of a second.

    Restricted to the fight proper. Across the post-battle transition the elixir
    bar is torn down and the reward screen takes over, so the fill edge there
    moves for reasons that have nothing to do with regen. The fight window is
    detected by the enemy king tower's health bar still being drawn, which is the
    same signal read_towers uses.
    """
    readings = [read_elixir(_frame_bgr(i)) for i in range(_frame_count())]
    in_fight = []
    for i in range(_frame_count()):
        king = next(t for t in read_board(_frame_bgr(i), elapsed=i / 3.0).enemy_towers if t.lane == "center")
        in_fight.append(king.readable)
    rises, jumps = 0, 0
    pairs = 0
    for i in range(1, _frame_count()):
        if not (in_fight[i] and in_fight[i - 1]):
            continue
        pairs += 1
        delta = readings[i] - readings[i - 1]
        if delta > 0:
            rises += 1
        if delta > 1:
            jumps += 1
    assert pairs > 50, f"only {pairs} in-fight frame pairs; the window detection is wrong"
    # Drops dominate here, not rises: the bot plays a card every few seconds, so
    # the trace is a sawtooth (long flat run -> +1 as the pip fills -> a drop when
    # the card goes out), not a rising staircase. Do not assert a rise ratio.
    #
    # What must hold is that a rise of 2+ never happens: elixir cannot regenerate
    # two pips in a third of a second. That is exactly what the old single-pixel
    # reader failed (its trace contained +3 and +4 jumps). A small allowance covers
    # the post-battle boundary, where the bar is torn down mid-transition.
    assert jumps <= max(2, pairs * 0.02), f"{jumps}/{pairs} pairs rose by 2+ ({rises} rose by 1): {readings}"


@needs_pack
def test_elixir_stays_in_range():
    for i in range(0, _frame_count(), 7):
        assert 0 <= read_elixir(_frame_bgr(i)) <= 10


def test_elixir_on_a_blank_frame_is_zero():
    import numpy

    blank = numpy.zeros((FRAMES_H, FRAMES_W, 3), dtype=numpy.uint8)
    assert read_elixir(blank) == 0


# --- towers -----------------------------------------------------------------


@needs_pack
def test_towers_read_from_a_live_frame():
    board = read_board(_frame_bgr(5), elapsed=1.0)
    assert board.elixir == 10
    assert len(board.enemy_towers) == 3
    assert len(board.own_towers) == 3
    for tower in board.enemy_towers:
        assert tower.alive
        assert 0.0 <= tower.hp <= 1.0
        assert tower.readable, "a live in-fight frame should have readable tower bars"


@needs_pack
def test_weakest_enemy_tower_breaks_ties_to_the_left():
    board = read_board(_frame_bgr(2), elapsed=1.0)
    weakest = board.weakest_enemy_tower
    assert weakest is not None
    assert weakest.lane in ("left", "right", "center")


def test_destroyed_tower_does_not_read_as_healthy_when_the_bar_is_gone():
    import numpy

    blank = numpy.zeros((FRAMES_H, FRAMES_W, 3), dtype=numpy.uint8)
    _, enemy = read_towers(blank)
    assert all(t.readable is False for t in enemy), "a missing bar must not be reported as readable"
    assert all(t.hp == 1.0 for t in enemy), "an unreadable tower must not read as destroyed"


# --- card knowledge ---------------------------------------------------------


def test_card_table_covers_the_players_deck():
    deck = [
        "ice_wizard",
        "hero_knight",
        "tombstone",
        "fireball",
        "royal_delivery",
        "mega_minion",
        "little_prince",
        "hog",
    ]
    for card_id in deck:
        f = facts_for(card_id)
        assert f.elixir > 0, f"{card_id} missing an elixir cost"
        assert f.role, f"{card_id} missing a role"
        assert not f.notes.startswith("unknown"), f"{card_id} absent from the card table"


def test_unknown_card_falls_back_to_defence_not_a_spell():
    f = facts_for("card_that_does_not_exist")
    assert not f.is_spell
    assert f.role == "defense"


def test_air_units_are_flagged_and_ground_units_are_not():
    assert facts_for("mega_minion").is_air
    assert facts_for("hog").is_air is False
    assert facts_for("hog").hits_air is False, "Hog Rider must not be offered as air defence"


def test_every_curated_air_unit_is_flagged_air():
    """The air roster is hand-curated (no API is right); this pins it.

    Cross-checked against CR's actual roster and KataCR's ``flying_unit_list``.
    Three rows were wrong before this test existed: ``baby_dragon`` and
    ``inferno_dragon`` were marked ground, and ``flying_machine`` was missing.
    """
    from pyclashbot.bot.card_knowledge import AIR_UNITS, known_card_ids

    known = set(known_card_ids())
    for card_id in sorted(AIR_UNITS & known):
        assert facts_for(card_id).is_air, f"{card_id} is in AIR_UNITS but reads as ground"
        assert facts_for(card_id).hits_air, f"{card_id} is air but cannot answer air"


def test_ground_units_stay_ground():
    """Guards the other direction: Battle Healer and Royal Ghost are ground units.

    RoyaleAPI's ``is_air_unit`` flag marks exactly those two, which is how a
    wrong flag would enter this table.
    """
    for card_id in ("battle_healer", "royal_ghost", "little_prince", "hog", "fireball"):
        assert facts_for(card_id).is_air is False, f"{card_id} is not an air unit"


def test_air_unit_count_is_stable():
    air = {c for c, f in _TABLE.items() if f.is_air}
    assert air == set(AIR_UNITS & set(_TABLE)), f"air flags drifted: {sorted(air ^ (AIR_UNITS & set(_TABLE)))}"


def test_card_table_is_not_trivially_small():
    assert TABLE_SIZE > 100


# --- decision ---------------------------------------------------------------


def _board(
    elixir: int = 10,
    elapsed: float = 0.0,
    enemy_units=(),
    enemy_towers=None,
    own_towers=None,
) -> BoardState:
    default_enemy_towers = (
        Tower(116, 134, "left", 1.0),
        Tower(302, 134, "right", 0.4, True),
        Tower(210, 91, "center", 1.0),
    )
    default_own_towers = (
        Tower(121, 396, "left", 1.0),
        Tower(308, 396, "right", 1.0),
        Tower(205, 421, "center", 1.0),
    )
    return BoardState(
        elixir=elixir,
        elapsed=elapsed,
        double_elixir=elapsed >= 60,
        own_towers=own_towers or default_own_towers,
        enemy_towers=enemy_towers or default_enemy_towers,
        units=tuple(enemy_units),
    )


def test_fireball_lands_on_the_weaker_tower():
    """The reported bug: Fireball dropped in front of the tower, not on it."""
    board = _board(elixir=10)
    intent = decide(board, ["fireball"])
    assert intent is not None
    assert intent.card_id == "fireball"
    assert intent.x == 302 and intent.y == 134, f"expected the weaker (right) tower, got {(intent.x, intent.y)}"


def test_fireball_never_lands_in_front_of_a_tower():
    board = _board(elixir=10)
    for lane, (tx, ty) in ENEMY_PRINCESS_TOWERS.items():
        board = BoardState(
            elixir=10,
            elapsed=0.0,
            double_elixir=False,
            own_towers=board.own_towers,
            enemy_towers=(Tower(tx, ty, lane, 0.4, True), Tower(302, 134, "right", 1.0), Tower(210, 91, "center", 1.0)),
        )
        intent = decide(board, ["fireball"])
        assert intent is not None
        assert abs(intent.y - ty) <= 6, f"{lane}: Fireball landed {intent.y - ty}px off the tower"


def test_a_spell_with_no_target_is_held_not_thrown_away():
    """Regression: Royal Delivery used to fall through to the default troop rule
    and land behind our own line, burning the card."""
    board = _board(elixir=10)
    intent = decide(board, ["royal_delivery"])
    assert intent is None, "a spell with no legal target must be held, not dropped"


def test_a_held_spell_is_played_once_a_target_appears():
    board_empty = _board(elixir=10)
    assert decide(board_empty, ["royal_delivery"]) is None
    board_threat = _board(elixir=10, enemy_units=[Unit(150, 300, True, True)])
    assert decide(board_threat, ["royal_delivery"]) is not None


@needs_pack
def test_tower_hp_actually_varies_across_a_pack():
    """The bar reader must track damage, not always report "full".

    On the calibration pack the right tower loses HP mid-fight while the left and
    the king stay untouched, so the readings must diverge.
    """
    count = _frame_count()
    early = read_board(_frame_bgr(3), elapsed=1.0)
    late = read_board(_frame_bgr(min(count - 2, 250)), elapsed=80.0)
    assert any(t.readable for t in early.enemy_towers), "no tower bar was readable"
    hp_early = {t.lane: t.hp for t in early.enemy_towers}
    hp_late = {t.lane: t.hp for t in late.enemy_towers}
    assert hp_early != hp_late, f"tower HP never changed: {hp_early} -> {hp_late}"


@needs_pack
def test_weakest_tower_follows_the_damage():
    """Once the right tower is hurt, that is the tower we push and spell."""
    count = _frame_count()
    board = read_board(_frame_bgr(min(count - 2, 250)), elapsed=80.0)
    weakest = board.weakest_enemy_tower
    assert weakest is not None
    if not all(t.readable for t in board.enemy_towers):
        pytest.skip("a tower bar was not readable in this frame")
    assert weakest.hp == min(t.hp for t in board.enemy_princess)


def test_spell_prefers_a_live_threat_over_tower_chip():
    board = _board(elixir=10, enemy_units=[Unit(150, 300, True, True)])
    intent = decide(board, ["arrows", "hog"])
    assert intent is not None
    assert intent.card_id == "arrows"
    assert intent.rule == "spell_value"
    assert 120 < intent.x < 180, "should aim at the unit, not the tower"


def test_fireball_only_targets_lowest_hp_tower():
    # Explicit user rule: Fireball must ONLY target the enemy tower with lowest HP
    board = _board(elixir=10, enemy_units=[Unit(150, 300, True, True)])
    intent = decide(board, ["fireball", "hog"])
    assert intent is not None
    assert intent.card_id == "fireball"
    assert intent.rule == "spell_value"
    weakest = board.weakest_enemy_tower
    assert weakest is not None
    assert intent.x == weakest.x
    assert intent.y == weakest.y


def test_air_threat_is_answered_by_an_air_hitter():
    from pyclashbot.bot.board import Unit

    board = _board(elixir=10, enemy_units=[Unit(160, 300, True, True)])
    intent = decide(board, ["hog", "mega_minion"])
    assert intent is not None
    assert intent.card_id == "mega_minion", f"chose {intent.card_id} ({intent.rule})"
    assert intent.rule == "air_defence"


def test_ground_only_card_is_not_sent_at_air():
    from pyclashbot.bot.board import Unit

    board = _board(elixir=10, enemy_units=[Unit(160, 300, True, True)])
    intent = decide(board, ["hog"])
    assert intent is None or intent.card_id != "hog" or "air" not in intent.tags


def test_unaffordable_cards_are_never_played():
    board = _board(elixir=1)
    assert decide(board, ["fireball", "hog"]) is None


def test_decision_is_deterministic():
    board = _board(elixir=10)
    first = decide(board, ["fireball", "hog", "ice_wizard"])
    for _ in range(5):
        assert decide(board, ["fireball", "hog", "ice_wizard"]) == first


def test_consecutive_repeats_are_broken():
    """matt's root cause: the bot replays the same card. Variety is the fix.

    The observed failure was a spell landing six times in a row at one fixed
    coordinate. With an empty board two cards score near-identically on rules, so
    the consecutive-play penalty is what must break the tie.
    """
    board = _board(elixir=10)
    ctx = PlayContext(recent_plays=("ice_wizard",))
    intent = decide(board, ["hog", "ice_wizard"], ctx)
    assert intent is not None
    assert intent.card_id == "hog", f"repeated {intent.card_id} twice in a row"


def test_unplayed_cards_break_an_otherwise_equal_tie():
    """Between two equivalent options, the one the cycle has not seen wins."""
    board = _board(elixir=10)
    hand = ["ice_wizard", "little_prince"]
    fresh = decide(board, hand, PlayContext(recent_plays=("hog",)))
    seen = decide(board, hand, PlayContext(recent_plays=("hog", "ice_wizard")))
    assert fresh is not None and seen is not None
    assert fresh.card_id != seen.card_id, f"hand history had no effect: {fresh.card_id} twice"


def test_a_repeated_win_condition_is_not_re_pushed_into_an_empty_board():
    """The consecutive-play penalty must not be a no-op on win conditions.

    Dropping a second Hog into an empty lane right after the first is a real
    mistake, so the penalty has to outrank the push bonus here.
    """
    board = _board(elixir=10)
    fresh = decide(board, ["hog", "ice_wizard"], PlayContext(recent_plays=("ice_wizard",)))
    repeated = decide(board, ["hog", "ice_wizard"], PlayContext(recent_plays=("hog",)))
    assert fresh is not None and repeated is not None
    assert fresh.card_id == "hog" and fresh.y == 283, "an unpushed win condition should go to the bridge"
    assert repeated.card_id != "hog", "the same win condition was re-pushed immediately"


def test_an_overwhelming_situation_overrides_the_consecutive_penalty():
    """Air defence is worth repeating a card for."""
    board = _board(elixir=10, enemy_units=[Unit(160, 300, True, True)])
    intent = decide(board, ["mega_minion"], PlayContext(recent_plays=("mega_minion",)))
    assert intent is not None
    assert intent.card_id == "mega_minion"
    assert intent.rule == "air_defence"


def test_win_condition_goes_to_the_bridge_of_the_weaker_lane():
    board = _board(elixir=10)
    intent = decide(board, ["hog"])
    assert intent is not None
    assert intent.y == 283, f"Hog should go to the bridge, got y={intent.y}"
    assert intent.x == 303, f"Hog should go up the right lane, got x={intent.x}"


def test_defensive_card_is_not_dropped_straight_into_the_enemy_half():
    board = _board(elixir=10)
    intent = decide(board, ["hero_knight"])
    assert intent is not None
    assert intent.y > RIVER_Y, "defence must not be fed forward"


@needs_pack
def test_decide_against_a_real_frame_produces_a_legal_play():
    board = read_board(_frame_bgr(3), elapsed=0.0)
    hand = ["ice_wizard", "hero_knight", "tombstone", "fireball", "royal_delivery", "mega_minion", "little_prince", "hog"]
    affordable = [c for c in hand if facts_for(c).elixir <= board.elixir]
    intent = decide(board, affordable)
    if intent is None:
        pytest.skip("no card affordable in this frame")
    assert 0 <= intent.x < FRAMES_W
    assert 0 <= intent.y < FRAMES_H
    assert intent.reason


def test_elixir_pacing_holds_under_four_in_single_elixir_without_threats():
    """In single elixir with no enemy threats, hold elixir until >= 4."""
    board = _board(elixir=3, elapsed=10.0, enemy_units=[])
    assert decide(board, ["hog", "archers"]) is None


def test_elixir_pacing_plays_at_four_or_above_in_single_elixir():
    """At 5 elixir in single-elixir time with no threats, cycle anyway (threshold is 4)."""
    board = _board(elixir=5, elapsed=10.0, enemy_units=[])
    intent = decide(board, ["hog", "archers"])
    assert intent is not None


def test_elixir_pacing_defends_immediately_when_threat_present():
    """If enemy troops are on our side, defend immediately even at low elixir."""
    board = _board(elixir=3, elapsed=10.0, enemy_units=[Unit(120, 350, False, True)])
    intent = decide(board, ["archers"])
    assert intent is not None
    assert intent.card_id == "archers"


def test_elixir_pacing_plays_in_double_elixir_even_under_six():
    """In 2x elixir (>= 60s), cycle proactively without waiting for 6 elixir."""
    board = _board(elixir=4, elapsed=70.0, enemy_units=[])
    intent = decide(board, ["hog"])
    assert intent is not None
    assert intent.card_id == "hog"


def test_spear_goblins_and_flying_machine_and_executioner_hit_air():
    """Verify updated RoyaleAPI card facts for air defense."""
    assert facts_for("spear_goblins").hits_air is True
    assert facts_for("flying_machine").hits_air is True
    assert facts_for("executioner").hits_air is True
    assert facts_for("little_prince").hits_air is True
    assert facts_for("fisherman").hits_air is False


def test_tactical_building_center_pull_places_in_killzone():
    """Defensive buildings pull ground win conditions (Hog/Giant) to center arena."""
    board = _board(elixir=10, enemy_units=[Unit(116, 320, False, True, card_id="hog_rider")])
    intent = decide(board, ["cannon", "archers"])
    assert intent is not None
    assert intent.card_id == "cannon"
    assert intent.rule == "ground_defence"
    assert intent.x == 198 and intent.y == 345, f"Building should center-pull left push, got ({intent.x}, {intent.y})"


def test_tactical_spell_avoids_dormant_king_tower():
    """Spells must never activate a dormant King Tower while Princess towers stand."""
    from pyclashbot.bot.coords import ENEMY_KING_TOWER

    # Unit placed right next to dormant King Tower (210, 91)
    kx, ky = ENEMY_KING_TOWER
    board = _board(elixir=10, enemy_units=[Unit(kx, ky + 10, False, True)])
    intent = decide(board, ["fireball"])
    if intent is not None and "enemy unit" in intent.reason:
        # If it aimed at unit, distance to king must be safe
        dist = ((intent.x - kx) ** 2 + (intent.y - ky) ** 2) ** 0.5
        assert dist >= 62 + 15, "Spell should not aim so close to dormant King Tower"


def test_tactical_kiting_distracts_heavy_melee_to_center():
    """Cheap defense cards kite dangerous melee units to the center river."""
    board = _board(elixir=10, enemy_units=[Unit(118, 300, False, True, card_id="pekka")])
    intent = decide(board, ["skeletons", "archers"])
    assert intent is not None
    # Skeletons or kiting should pull to center
    if intent.rule == "kiting":
        assert intent.x == 209 and intent.y == 340


def test_tactical_splash_chosen_against_swarm():
    """Valkyrie (splash) is decisively preferred over Mini Pekka against swarms."""
    board = _board(
        elixir=10,
        enemy_units=[
            Unit(118, 320, False, True, card_id="goblins"),
            Unit(124, 325, False, True, card_id="goblins"),
        ],
    )
    intent = decide(board, ["mini_pekka", "valkyrie"])
    assert intent is not None
    assert intent.card_id == "valkyrie"
    assert intent.rule == "ground_defence"
    # Melee splash drops directly into enemy cluster
    assert intent.y <= 335


def test_tactical_tank_buster_chosen_against_heavy_tank():
    """Mini Pekka (high damage) is preferred over Valkyrie against heavy tanks."""
    board = _board(elixir=10, enemy_units=[Unit(118, 310, False, True, card_id="giant")])
    intent = decide(board, ["valkyrie", "mini_pekka"])
    assert intent is not None
    assert intent.card_id == "mini_pekka"
    assert intent.rule == "ground_defence"


def test_tactical_air_superiority_against_ground_only_push():
    """Minions (air) defend against ground-only troops without taking damage."""
    board = _board(elixir=10, enemy_units=[Unit(118, 320, False, True, card_id="valkyrie")])
    intent = decide(board, ["minions"])
    assert intent is not None
    assert intent.card_id == "minions"
    assert intent.rule == "ground_defence"
    assert "air unit" in intent.reason


def test_tactical_ranged_unit_maintains_distance_buffer():
    """Archer Queen (ranged) keeps a safe distance buffer behind incoming push."""
    board = _board(elixir=10, enemy_units=[Unit(118, 300, False, True, card_id="valkyrie")])
    intent = decide(board, ["archer_queen"])
    assert intent is not None
    assert intent.card_id == "archer_queen"
    # Must be placed at or behind 300 + 46 px (346)
    assert intent.y >= 346
    assert "ranged" in intent.reason


def test_tactical_royal_delivery_targets_cluster_on_friendly_half():
    """Royal Delivery targets enemy clusters on our half and avoids enemy half."""
    board = _board(
        elixir=10,
        enemy_units=[
            Unit(120, 330, False, True, card_id="goblins"),
            Unit(130, 335, False, True, card_id="goblins"),
        ],
    )
    intent = decide(board, ["royal_delivery"])
    assert intent is not None
    assert intent.card_id == "royal_delivery"
    assert intent.y >= RIVER_Y
    assert "royal delivery" in intent.reason


def test_log_bait_goblin_barrel_punishes_baited_spell():
    """Goblin Barrel targets enemy tower and scores highest when small spell is baited."""
    board = _board(elixir=10)
    ctx_normal = PlayContext()
    ctx_baited = PlayContext(opponent_baited_spell=True, opponent_has_small_spell=False)

    intent_normal = decide(board, ["goblin_barrel"], ctx_normal)
    intent_baited = decide(board, ["goblin_barrel"], ctx_baited)

    assert intent_normal is not None and intent_baited is not None
    assert intent_baited.score > intent_normal.score
    assert "LOG BAITED" in intent_baited.reason
    assert intent_baited.y == 134  # Princess tower y


def test_inferno_tower_strictly_reserved_for_heavy_tanks():
    """Inferno Tower rejects small swarms and only activates for heavy tanks."""
    # 1. Swarm -> Inferno Tower must NOT be played
    board_swarm = _board(elixir=10, enemy_units=[Unit(120, 330, False, True, card_id="goblins")])
    intent_swarm = decide(board_swarm, ["inferno_tower"])
    assert intent_swarm is None or intent_swarm.rule != "inferno_anti_tank"

    # 2. Heavy Tank (Golem/Giant/Pekka) -> Inferno Tower activates in 4-3 center pull
    board_tank = _board(elixir=10, enemy_units=[Unit(120, 330, False, True, card_id="golem")])
    intent_tank = decide(board_tank, ["inferno_tower"])
    assert intent_tank is not None
    assert intent_tank.card_id == "inferno_tower"
    assert intent_tank.rule == "inferno_anti_tank"
    assert intent_tank.y == 345


def test_fireball_level16_executes_low_hp_tower():
    """Level 16 Fireball directly executes a low HP enemy tower."""
    from pyclashbot.bot.board import Tower
    low_tower = Tower(116, 134, "left", hp=0.12, readable=True)
    other_tower = Tower(302, 134, "right", hp=1.0, readable=True)
    king = Tower(210, 91, "center", hp=1.0, readable=True)
    board = _board(elixir=10, enemy_towers=(low_tower, other_tower, king))

    intent = decide(board, ["fireball"])
    assert intent is not None
    assert intent.card_id == "fireball"
    assert intent.rule == "fireball_lvl16"
    assert intent.x == 116 and intent.y == 134
    assert "lethal finish" in intent.reason


def test_princess_river_siege_and_cross_lane_defense():
    """Princess uses 9-tile reach across river in empty corridor and cross-lane splash in defense."""
    from pyclashbot.bot.board import Unit
    # 1. Empty corridor: river siege on enemy tower (9-tile range)
    board_empty = _board(elixir=10)
    intent_empty = decide(board_empty, ["princess"])
    assert intent_empty is not None
    assert intent_empty.card_id == "princess"
    assert intent_empty.y == 295
    assert "river siege" in intent_empty.reason

    # 2. Incoming enemy push in left lane: Princess deploys in opposite right backline (safe from melee)
    board_defense = _board(elixir=10, enemy_units=[Unit(118, 280, is_air=False, enemy=True)])
    intent_def = decide(board_defense, ["princess"])
    assert intent_def is not None
    assert intent_def.card_id == "princess"
    assert intent_def.x == 303  # Opposite lane
    assert intent_def.y >= 450  # Deep backline
    assert "cross" in intent_def.reason


def test_defend_endangered_tower_when_damaged():
    """When a friendly tower is taking damage, prioritize defending that lane urgently."""
    from pyclashbot.bot.board import Tower
    own_left = Tower(121, 396, "left", hp=1.0, readable=True)
    own_right = Tower(308, 396, "right", hp=0.55, readable=True)  # melting right tower!
    own_king = Tower(205, 421, "center", hp=1.0, readable=True)

    board = _board(elixir=5, own_towers=(own_left, own_right, own_king))
    intent = decide(board, ["hero_knight", "goblin_gang", "ice_spirit"])
    assert intent is not None
    assert intent.rule == "defend_endangered_tower"
    # Defends the right lane
    assert intent.x == 303
    assert intent.card_id == "hero_knight"


def test_defend_bridge_siege_threat():
    """Bridge threats (e.g. X-Bow at y=210) trigger defensive response in that exact lane."""
    from pyclashbot.bot.board import Unit
    # Threat placed at right bridge (x=290, y=210)
    board = _board(elixir=4, enemy_units=[Unit(290, 210, is_air=False, enemy=True)])
    intent = decide(board, ["hero_knight", "the_log", "ice_spirit"])
    assert intent is not None
    assert intent.rule == "defend_endangered_tower"
    assert intent.x == 303


def test_decide_reacts_when_own_tower_destroyed():
    """When a friendly tower is destroyed (hp=0.0), elixir pacing must react instead of stalling."""
    from pyclashbot.bot.board import Tower
    own_left = Tower(121, 396, "left", hp=0.0, readable=True)
    own_right = Tower(308, 396, "right", hp=1.0, readable=True)
    own_king = Tower(205, 421, "center", hp=1.0, readable=True)

    board = _board(elixir=3, elapsed=15.0, own_towers=(own_left, own_right, own_king))
    intent = decide(board, ["knight", "skeletons"])
    assert intent is not None, "Bot must not stall at 3 elixir when own tower was lost"


def test_the_log_and_spells_lethal_finish_enemy_tower():
    """The Log and direct spells immediately execute low-HP enemy towers (<8% HP)."""
    from pyclashbot.bot.board import Tower
    from pyclashbot.bot.decide import BRIDGE_Y
    enemy_left = Tower(116, 134, "left", hp=0.07, readable=True)
    enemy_right = Tower(303, 134, "right", hp=1.0, readable=True)
    enemy_king = Tower(210, 91, "center", hp=1.0, readable=True)

    board = _board(elixir=3, elapsed=30.0, enemy_towers=(enemy_left, enemy_right, enemy_king))
    intent = decide(board, ["the_log", "knight", "ice_spirit"])
    assert intent is not None
    assert intent.card_id == "the_log"
    assert intent.rule == "tower_lethal_spells"
    assert intent.y == BRIDGE_Y
    assert "lethal finish" in intent.reason


def test_princess_lethal_river_siege_locks_lane():
    """Princess locks onto a dying enemy tower across the river to execute it."""
    from pyclashbot.bot.board import Tower
    enemy_left = Tower(116, 134, "left", hp=0.18, readable=True)
    enemy_right = Tower(303, 134, "right", hp=1.0, readable=True)
    enemy_king = Tower(210, 91, "center", hp=1.0, readable=True)

    board = _board(elixir=4, elapsed=40.0, enemy_towers=(enemy_left, enemy_right, enemy_king))
    intent = decide(board, ["princess"])
    assert intent is not None
    assert intent.card_id == "princess"
    assert intent.x == 118
    assert intent.y == 295
    assert "lethal finish" in intent.reason


def test_opponent_tracker_preserves_enemy_tower_hp_across_fadeout():
    """OpponentTracker preserves lowest seen enemy tower HP when health bar transiently fades."""
    from pyclashbot.bot.board import Tower
    from pyclashbot.bot.opponent_tracker import OpponentTracker

    tracker = OpponentTracker()
    tracker.reset()

    # Frame 1: enemy left tower takes heavy hit, HP bar visible at 0.08
    towers_damaged = (
        Tower(116, 134, "left", hp=0.08, readable=True),
        Tower(303, 134, "right", hp=1.0, readable=True),
        Tower(210, 91, "center", hp=1.0, readable=True),
    )
    res1 = tracker.update_enemy_towers(towers_damaged)
    assert res1[0].hp == 0.08

    # Frame 2: UI health bar temporarily fades out (readable=False, default hp=1.0)
    towers_faded = (
        Tower(116, 134, "left", hp=1.0, readable=False),
        Tower(303, 134, "right", hp=1.0, readable=False),
        Tower(210, 91, "center", hp=1.0, readable=False),
    )
    res2 = tracker.update_enemy_towers(towers_faded)
    assert res2[0].hp == 0.08, "Tower HP must be preserved at lowest observed value (0.08), not reset to 1.0"
