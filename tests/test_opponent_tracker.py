"""Tests for the opponent elixir and card cycle tracker."""

from pyclashbot.bot.opponent_tracker import OpponentTracker


def test_tracker_initializes_at_starting_elixir():
    tracker = OpponentTracker()
    assert tracker.current_elixir == 5.0


def test_tracker_regenerates_elixir_over_time():
    tracker = OpponentTracker()
    tracker.last_update_time -= 5.6  # 5.6s elapsed in single elixir
    tracker.update(elapsed=10.0, units=[])
    # 5.6s at 2.8s per elixir should add ~2.0 elixir
    assert 6.9 <= tracker.current_elixir <= 7.1


def test_tracker_caps_at_max_elixir():
    tracker = OpponentTracker()
    tracker.last_update_time -= 100.0
    tracker.update(elapsed=10.0, units=[])
    assert tracker.current_elixir == 10.0


def test_tracker_deducts_elixir_on_enemy_card_play():
    tracker = OpponentTracker(current_elixir=8.0)
    # Opponent plays Hog Rider (4 elixir)
    tracker.on_opponent_play("hog_rider")
    assert tracker.current_elixir == 4.0
    assert tracker.played_history[-1] == "hog"


def test_tracker_card_cycle_rotation():
    tracker = OpponentTracker()
    # Opponent plays Hog Rider
    tracker.on_opponent_play("hog_rider")
    assert not tracker.is_card_in_hand("hog_rider")

    # Opponent plays 3 more cards (cycle length is 4)
    tracker.on_opponent_play("fireball")
    tracker.on_opponent_play("ice_spirit")
    tracker.on_opponent_play("skeletons")
    assert not tracker.is_card_in_hand("hog_rider")

    # Opponent plays a 4th other card -> Hog Rider returns to hand!
    tracker.on_opponent_play("musketeer")
    assert tracker.is_card_in_hand("hog_rider")


def test_tracker_detects_units_and_tracks_win_condition():
    from pyclashbot.bot.board import Unit

    tracker = OpponentTracker(current_elixir=10.0)
    # Opponent drops Balloon (5 elixir wincon) detected by YOLO with enemy_ prefix
    balloon_unit = Unit(x=120, y=250, is_air=True, enemy=True, card_id="enemy_balloon")
    tracker.update(elapsed=30.0, units=[balloon_unit])

    # Elixir should be deducted by 5
    assert tracker.current_elixir <= 5.1
    assert tracker.opponent_win_condition == "balloon"
    assert not tracker.is_opponent_wincon_in_hand()

    # Opponent cycles 4 other cards
    for card in ["bomber", "guards", "arrows", "ice_spirit"]:
        tracker.on_opponent_play(card)

    assert tracker.is_opponent_wincon_in_hand()
