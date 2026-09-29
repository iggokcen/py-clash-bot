"""random module for randomizing fight plays"""

import collections
import dataclasses
import random
import time
from typing import Literal

from pyclashbot.bot import card_detection
from pyclashbot.bot.board import read_board
from pyclashbot.bot.card_detection import (
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_play_coords_for_card,
    identify_available_hand,
    is_hero_champion_ability_visible,
    switch_side,
    trigger_hero_champion_ability,
)
from pyclashbot.bot.coords import (
    CLOSE_BATTLE_LOG_BUTTON,
    EMOTE_BUTTON_COORD,
    EMOTE_ICON_COORDS,
    HAND_CARDS_COORDS,
    PLAYABLE_PLAY_REGION_LTRB,
    QUICKMATCH_POPUP_BUTTON_COORD,
    START_FIGHT_BUTTON_COORD,
)
from pyclashbot.bot.decide import PlayContext, decide
from pyclashbot.bot.nav import (
    check_for_in_battle_with_delay,
    get_to_activity_log,
    get_to_main_after_fight,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.bot.opponent_tracker import OpponentTracker
from pyclashbot.bot.recorder import (
    finish_fight_recording,
    is_recording,
    log_play,
    start_fight_recording,
    stop_fight_capture,
)
from pyclashbot.bot.state_detect import (
    check_if_battle_has_ended,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    check_pixels_for_win_in_battle_log,
    count_elixir,
)
from pyclashbot.utils.logger import Logger
from pyclashbot.utils.versioning import __version__

ELIXIR_WAIT_TIMEOUT = 40  # too high but someone got errors with that so idk
ABILITY_TRIGGER_DELAY_S = 1.0
_hero_ability_available_since: float | None = None


def _check_and_maybe_trigger_hero_ability(emulator, logger: Logger) -> bool:
    global _hero_ability_available_since
    if is_hero_champion_ability_visible(emulator):
        if _hero_ability_available_since is None:
            _hero_ability_available_since = time.time()
            logger.change_status(
                f"Hero/Champion ability ready — triggering in {ABILITY_TRIGGER_DELAY_S}s",
            )
        elif time.time() - _hero_ability_available_since >= ABILITY_TRIGGER_DELAY_S:
            trigger_hero_champion_ability(emulator, logger)
            _hero_ability_available_since = None
            return True
    else:
        _hero_ability_available_since = None
    return False


def _maybe_start_fight_recording(
    emulator, logger, recording_flag: bool, fight_mode_chosen: str, custom_path: str | None = None
) -> None:
    """Begin opt-in training-data capture for 1v1-type fights only (Trophy Road / Classic 1v1, never 2v2)."""
    if recording_flag and fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
        pack_mode = "1v1_trophy" if fight_mode_chosen == "Trophy Road" else "1v1_classic"
        start_fight_recording(emulator, pack_mode, __version__, logger=logger, custom_path=custom_path)


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    fight_mode_chosen,
    called_from_launching=False,
    recording_flag: bool = False,
    custom_path: str | None = None,
    observer_mode: bool = False,
) -> bool:
    """Handle the entirety of a battle state (start fight, do fight, end fight)."""

    logger.change_status("Waiting for battle to start")

    # Wait for battle start
    if wait_for_battle_start(emulator, logger) is False:
        logger.change_status("Timed out waiting for battle to start")
        return False

    logger.change_status("Starting fight loop")
    logger.log(f'This is the fight mode: "{fight_mode_chosen}"')
    if observer_mode:
        logger.log("[OBSERVER] Live match observer mode is ENABLED")

    # Recording is started/stopped inside the fight loops themselves so each pack
    # brackets exactly the in-battle play loop (no pre-battle wait or post-fight nav).
    # Run regular fight loop if random mode not toggled
    if (
        not random_fight_mode
        and _fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path, observer_mode=observer_mode)
        is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Run random fight loop if random mode toggled
    if (
        random_fight_mode
        and _random_fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Only log the fight if not called from the start
    if not called_from_launching:
        if fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_2v2_fights()

        if fight_mode_chosen == "Trophy Road":
            logger.increment_trophy_road_fights()
        elif fight_mode_chosen == "Classic 1v1":
            logger.increment_classic_1v1_fights()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_classic_2v2_fights()

    time.sleep(10)
    return True


def start_fight(emulator, logger, mode) -> bool:
    """Start a fight with the specified mode.

    Args:
        emulator: The emulator controller
        logger: Logger instance
        mode: Fight mode - must be one of "Classic 1v1", "Classic 2v2", or "Trophy Road"

    Returns:
        bool: True if fight started successfully, False otherwise
    """
    # Validate mode parameter
    logger.log(f'Input mode type: "{type(mode)}"')
    logger.log(f"Input mode value: {mode}")
    valid_modes = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    logger.log(f"Valid modes: {valid_modes}")
    if mode not in valid_modes:
        logger.log(f"The valid modes for start_fight() are: {valid_modes}")
        logger.log(f"But start_fight() got an invalid mode: '{mode}'")
        return False

    logger.change_status(f"Starting a {mode} fight")

    # Check if on clash main menu
    logger.log("Checking if on main menu before starting fight...")
    if not check_if_on_clash_main_menu(emulator):
        logger.change_status("Not on main menu — cannot start fight")
        return False

    # For all modes (1v1 and 2v2), use the same start button
    # Mode is already set by select_mode() in states.py, just click start button
    emulator.click(START_FIGHT_BUTTON_COORD[0], START_FIGHT_BUTTON_COORD[1])
    logger.log(f"Clicked Start button at {START_FIGHT_BUTTON_COORD}")

    # 2v2 needs a second popup after Start
    if mode == "Classic 2v2":
        logger.change_status("Classic 2v2 — clicking Quick Match popup...")
        time.sleep(3)
        emulator.click(QUICKMATCH_POPUP_BUTTON_COORD[0], QUICKMATCH_POPUP_BUTTON_COORD[1])
        logger.log(f"Clicked Quickmatch button at {QUICKMATCH_POPUP_BUTTON_COORD}")

    return True


def send_emote(emulator, logger: Logger):
    """Method to do an emote in a fight"""
    logger.change_status("Sending emote")

    # click emote button
    emulator.click(EMOTE_BUTTON_COORD[0], EMOTE_BUTTON_COORD[1])
    time.sleep(0.33)

    emote_coord = random.choice(EMOTE_ICON_COORDS)
    emulator.click(emote_coord[0], emote_coord[1])


RANDOM_PLAY_ELIXIR_MIN = 3
RANDOM_PLAY_ELIXIR_MAX = 9


def play_random_available_card(emulator, logger, recording_flag: bool, elapsed_s: float) -> bool:
    """Play one card that is actually available, at a random friendly-half coord.

    Mirrors the rl-bot RandomPlayer so recorded plays are never polluted with cards
    that didn't deploy: read the available hand slots (affordable, non-empty) and, if
    any, play a random one; if none are available, do nothing. Returns True if a card
    was played. The caller gates this behind a random elixir wait.
    """
    card_indices = check_which_cards_are_available(emulator, check_side=True)
    if not card_indices:
        return False  # nothing playable -> no play, nothing recorded (caller re-rolls)

    card_index = random.choice(card_indices)
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    play_coord = (random.randint(left, right), random.randint(top, bottom))

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    time.sleep(0.1)
    emulator.click(play_coord[0], play_coord[1])
    time.sleep(0.1)

    if recording_flag:
        # Card identity isn't computed for random plays; home re-derives it from pixels.
        log_play(card_index, play_coord[0], play_coord[1], elapsed_s)
    logger.add_card_played()
    return True


def wait_for_elixir(
    emulator,
    logger,
    elixir_wait_amount,
    WAIT_THRESHOLD=5000,  # noqa: N803
    PLAY_THRESHOLD=10000,  # noqa: N803
    recording_flag: bool = False,
) -> Literal["restart", "no battle"] | bool:
    """Method to wait for 4 elixir during a battle"""
    start_time = time.time()
    battle_detection_lost_count = 0
    last_logged_second = -1
    last_lost_detection_log_second = -1

    while not count_elixir(emulator, elixir_wait_amount):
        # debug screenshot saving removed from production
        wait_time = time.time() - start_time
        elapsed_second = int(wait_time)
        if elapsed_second != last_logged_second:
            logger.change_status(
                f"Waiting for {elixir_wait_amount} elixir for {elapsed_second}s...",
            )
            last_logged_second = elapsed_second

        card_indices = check_which_cards_are_available(emulator)
        _check_and_maybe_trigger_hero_ability(emulator, logger)

        card_inhand = len(card_indices)
        action_offset, _ = switch_side()
        if action_offset > PLAY_THRESHOLD and card_inhand > 0:
            logger.change_status("Battle too active — playing now")
            return True

        if action_offset > WAIT_THRESHOLD and card_inhand == 4:
            logger.change_status("All cards are available!")
            return True

        if wait_time > ELIXIR_WAIT_TIMEOUT:
            logger.change_status(status="Waited too long for elixir")
            return "restart"

        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status(status="Battle ended — stopping elixir wait")
                return "no battle"

            battle_detection_lost_count += 1
            lost_detection_second = int(time.time())
            if lost_detection_second != last_lost_detection_log_second:
                logger.change_status(
                    status="Lost battle detection while waiting for elixir — assuming still in battle",
                )
                last_lost_detection_log_second = lost_detection_second
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    status="Lost battle detection repeatedly — assuming battle ended",
                )
                return "no battle"

            time.sleep(0.5)
            continue

        battle_detection_lost_count = 0

    logger.change_status(
        f"Took {str(time.time() - start_time)[:4]}s for {elixir_wait_amount} elixir.",
    )

    return True


def end_fight_state(
    emulator,
    logger: Logger,
    recording_flag,
    disable_win_tracker_toggle=True,
):
    """Method to handle the time after a fight and before the next state"""
    # count the crown score on this end-battle screen

    # get to clash main after this fight
    logger.log("Returning to main menu after fight")
    if get_to_main_after_fight(emulator, logger) is False:
        logger.log("Failed to return to main menu after fight")
        finish_fight_recording(None)
        return False

    logger.log("Returned to main menu after fight")
    time.sleep(3)

    # Determine the outcome. Force the win check when a recording is active so the
    # pack gets a real win/loss; otherwise honor the user's win-tracker toggle.
    if is_recording() or not disable_win_tracker_toggle:
        win_check_return = check_if_previous_game_was_win(emulator, logger)

        if win_check_return == "restart":
            logger.log("Failed while checking if previous game was a win")
            finish_fight_recording(None)
            return False

        outcome = "win" if win_check_return else "loss"

        # Trigger online learning policy update on match outcome
        trainer = getattr(logger, "policy_trainer", None)
        if trainer is not None:
            reward = 1.0 if outcome == "win" else -1.0
            try:
                trainer.update(reward)
            except Exception as e:
                logger.log(f"Policy match update error: {e}")

        # Only touch the user's win/loss stats when their tracker is enabled.
        if not disable_win_tracker_toggle:
            if win_check_return:
                logger.add_win()
            else:
                logger.add_loss()

        finish_fight_recording(outcome)
    else:
        logger.log("Not checking win/loss because check is disabled")
        finish_fight_recording(None)

    return True


def check_if_previous_game_was_win(
    emulator,
    logger: Logger,
) -> bool | Literal["restart"]:
    """Method to handle the checking if the previous game was a win or loss"""
    logger.change_status(status="Checking last game result")

    # Use wait_for_clash_main_menu to ensure we are on the main menu.
    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        logger.change_status(status="Not on main menu — cannot check last game result")
        return "restart"

    # get to clash main options menu
    if get_to_activity_log(emulator, logger, printmode=False) == "restart":
        logger.change_status(status="Failed to open battle log")

        return "restart"

    logger.change_status(status="Checking battle log for win...")
    is_a_win = check_pixels_for_win_in_battle_log(emulator)
    result = "win" if is_a_win else "loss"
    logger.change_status(status=f"Last game result: {result}")

    # close battle log
    logger.change_status(status="Returning to main menu")
    emulator.click(CLOSE_BATTLE_LOG_BUTTON[0], CLOSE_BATTLE_LOG_BUTTON[1])
    if wait_for_clash_main_menu(emulator, logger) is False:
        logger.change_status(status="Timed out returning to main menu after battle log")
        return "restart"
    time.sleep(2)

    return is_a_win


# Initialize deques to store recent plays
last_three_cards = collections.deque(maxlen=3)
recent_card_ids: collections.deque[str] = collections.deque(maxlen=4)
opponent_tracker = OpponentTracker()


def select_card_index(card_indices, last_three_cards):
    if not card_indices:
        raise ValueError("card_indices cannot be empty")

    # First preference: Cards not in the last_three_cards queue
    preferred_cards = [index for index in card_indices if index not in last_three_cards]

    # Second preference: Cards not among the last two added to the queue
    if not preferred_cards and len(last_three_cards) == 3:
        preferred_cards = [index for index in card_indices if index not in list(last_three_cards)[-2:]]

    # Third preference: Any card except the most recently added one
    if not preferred_cards and last_three_cards:
        preferred_cards = [index for index in card_indices if index != last_three_cards[-1]]

    # Fallback: If all else fails, consider all cards
    if not preferred_cards:
        preferred_cards = card_indices

    return random.choice(preferred_cards)


def _strategy_choice(logger, battle_strategy: "BattleStrategy", card_indices):
    """Ask the strategy engine for a play, from the frame already captured.

    ``check_which_cards_are_available`` has just populated ``battle_iar``, so the
    hand, the elixir count and the board all come out of that one screenshot --
    being smarter costs no extra captures.

    Returns ``(card_index, card_id, coord)``, or ``None`` when the engine has no
    opinion, in which case the caller falls back to the legacy random placement.
    A fight must never stall because the board could not be read.
    """
    # Read the module attribute, not an imported copy: check_which_cards_are_available
    # rebinds battle_iar on every call, so a from-import would freeze it at None.
    frame = card_detection.battle_iar
    if frame is None:
        return None
    try:
        hand = identify_available_hand(card_indices)
        if not hand:
            return None
        elapsed = battle_strategy.get_elapsed_time()
        board = read_board(frame, elapsed)

        # Augment board units with YOLO ONNX detector when available
        detector = getattr(logger, "_field_detector", None)
        if detector is not None and frame is not None:
            try:
                from pyclashbot.bot.board import Unit
                from pyclashbot.detection.onnx_detector import split_detections_by_side

                dets = detector.detect(frame)
                enemy_dets, ally_dets, _ = split_detections_by_side(frame, dets)
                yolo_units = []
                for d in enemy_dets:
                    cx = int((d.x1 + d.x2) // 2)
                    cy = int((d.y1 + d.y2) // 2)
                    is_air = bool(d.name and ("bat" in d.name or "dragon" in d.name or "balloon" in d.name or "minion" in d.name))
                    yolo_units.append(Unit(x=cx, y=cy, is_air=is_air, enemy=True, card_id=d.bot_id or d.name))
                for d in ally_dets:
                    cx = int((d.x1 + d.x2) // 2)
                    cy = int((d.y1 + d.y2) // 2)
                    is_air = bool(d.name and ("bat" in d.name or "dragon" in d.name or "balloon" in d.name or "minion" in d.name))
                    yolo_units.append(Unit(x=cx, y=cy, is_air=is_air, enemy=False, card_id=d.bot_id or d.name))
                if yolo_units:
                    board = dataclasses.replace(board, units=tuple(yolo_units))
            except Exception:
                pass

        tracked_enemy_towers = opponent_tracker.update_enemy_towers(board.enemy_towers)
        board = dataclasses.replace(board, enemy_towers=tracked_enemy_towers)
        opponent_tracker.update(elapsed, board.units)
        opp_elixir = opponent_tracker.current_elixir
        wincon = opponent_tracker.opponent_win_condition
        wincon_str = f" | WinCon: {wincon}" if wincon else ""
        hand_status = " (IN HAND)" if opponent_tracker.is_opponent_wincon_in_hand() else " (In Cycle)"
        low_elixir_str = " [LOW ELIXIR - ATTACK!]" if opponent_tracker.is_low_elixir() else ""
        logger.change_status(f"HUD: Opponent: {opp_elixir:.1f}/10 elixir{wincon_str}{hand_status if wincon else ''}{low_elixir_str}")
        hero_knight_present = any(
            "knight" in (getattr(u, "card_id", "") or "").lower()
            for u in board.units
            if not getattr(u, "enemy", True)
        )
        intent = decide(
            board,
            list(hand.values()),
            PlayContext(
                recent_plays=tuple(recent_card_ids),
                opponent_is_low_elixir=opponent_tracker.is_low_elixir(),
                opponent_has_small_spell=opponent_tracker.has_small_spell_in_hand(),
                opponent_baited_spell=opponent_tracker.recently_baited_small_spell(),
                hero_knight_on_field=hero_knight_present,
                opponent_win_condition=wincon,
            ),
        )
    except Exception:
        logger.log("Strategy engine could not decide; falling back to legacy placement")
        return None
    if intent is None:
        return "WAIT"
    index = next((i for i, card_id in hand.items() if card_id == intent.card_id), None)
    if index is None:
        # The engine named a card we could not fingerprint in this frame, or the
        # same id appears twice in hand. Take the first playable slot rather than
        # clicking a card we did not choose.
        index = card_indices[0]
    logger.change_status(
        f"Strategy: {intent.card_id} via {intent.rule} — {intent.reason} (score {intent.score:.0f})",
    )
    return index, hand.get(index, "unknown"), (intent.x, intent.y)


def play_a_card(emulator, logger, recording_flag: bool, battle_strategy: "BattleStrategy") -> bool:
    print("\n")

    # check which cards are available
    logger.change_status("Looking at which cards are available")
    available_card_check_start_time = time.time()
    card_indices = check_which_cards_are_available(emulator, check_side=True)

    if not card_indices:
        logger.change_status("No cards ready yet...")
        return False

    available_card_check_time_taken = str(
        time.time() - available_card_check_start_time,
    )[:3]

    logger.change_status(
        f"These cards are available: {card_indices} ({available_card_check_time_taken}s)",
    )

    _check_and_maybe_trigger_hero_ability(emulator, logger)
    choice = _strategy_choice(logger, battle_strategy, card_indices)
    if choice == "WAIT":
        # Override WAIT if enough cards are available to prevent the bot from
        # stalling forever when board/threat detection is unreliable (e.g. new
        # accounts with unknown cards).  3+ cards is a strong enough hand.
        if len(card_indices) >= 3:
            logger.change_status(f"Overriding WAIT ({len(card_indices)}/4 cards available) — cycling to prevent stall")
            card_index = select_card_index(card_indices, last_three_cards)
            card_id = None
            play_coord = None
        else:
            logger.change_status(
                f"Strategy engine returned WAIT (pooling elixir: {len(card_indices)}/4 cards available)",
            )
            return True
    elif choice is not None:
        card_index, card_id, play_coord = choice
    else:
        card_index = select_card_index(card_indices, last_three_cards)
        card_id = None
        play_coord = None
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)
    if card_id and card_id != "unknown":
        recent_card_ids.append(card_id)
    logger.change_status(f"Choosing this card index: {card_index}")

    if play_coord is None:
        # Legacy placement: card group -> fixed zone coords, side chosen by the
        # bridge detector. Kept as the fallback so an engine read failure
        # degrades to the old behaviour rather than to no behaviour.
        play_coord_calculation_start_time = time.time()
        card_id, play_coord = get_play_coords_for_card(emulator, logger, card_index, battle_strategy.get_elapsed_time())
        play_coord_calculation_time_taken = str(
            time.time() - play_coord_calculation_start_time,
        )[:3]

        logger.change_status(
            f"Calculated play for: {card_id} at {play_coord} ({play_coord_calculation_time_taken}s)",
        )

    # click the card index
    click_and_play_card_start_time = time.time()
    if None in [HAND_CARDS_COORDS, card_index]:
        logger.change_status("Non-fatal error: card index is None")
        return False

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])

    # click the play coord
    if play_coord is None:
        logger.change_status("Non-fatal error: play coordinates are None")
        return False

    emulator.click(play_coord[0], play_coord[1])
    click_and_play_card_time_taken = str(time.time() - click_and_play_card_start_time)[:3]
    if recording_flag:
        log_play(card_index, play_coord[0], play_coord[1], battle_strategy.get_elapsed_time())

    logger.change_status(f"Made the play {click_and_play_card_time_taken}s")
    logger.add_card_played()

    # Step update for online learning policy
    trainer = getattr(logger, "policy_trainer", None)
    if trainer is not None and play_coord is not None:
        try:
            from pyclashbot.bot.card_detection import get_card_group
            from pyclashbot.bot.policy import PolicyAction, build_state_vector

            card_grp = get_card_group(card_id or "unknown")
            action = PolicyAction(
                card_index=card_index,
                card_group=card_grp,
                coord=play_coord,
                card_id=card_id,
            )
            side_flag = "right" if play_coord[0] >= 210 else "left"
            avail_mask = [1 if i in card_indices else 0 for i in range(4)]
            imm_reward = 0.05
            if play_coord[1] >= 340 and card_grp not in ("spell", "ground_spell", "large_spell"):
                imm_reward += 0.05
            state_vec = build_state_vector(
                elapsed_time=battle_strategy.get_elapsed_time(),
                elixir=4,
                side_preference=side_flag,
                available_mask=avail_mask,
            )
            trainer.update_step(state_vec, action, imm_reward)
            trainer.record(
                state_vec,
                action,
                elapsed_time=battle_strategy.get_elapsed_time(),
            )
        except Exception:
            pass

    if random.randint(0, 9) == 1:
        send_emote(emulator, logger)
    return True


_PHASE_STRATEGIES = {
    "early": [0.55, 0.35, 0.10, 0, 0, 0, 0],  # 0-7s: Fast opening, start at 3-4 elixir immediately
    "single": [0.45, 0.40, 0.15, 0, 0, 0, 0],  # 7-90s: Active 3-5 elixir cycling to maintain pressure
    "double": [0.40, 0.40, 0.20, 0, 0, 0, 0],  # 90-200s: Rapid attack pressure
    "triple": [0.40, 0.40, 0.20, 0, 0, 0, 0],  # 200s+: Continuous assault
}

_PHASE_THRESHOLDS_1V1 = {
    "early": (1500, 3000),
    "single": (2000, 4000),
    "double": (1500, 3000),
    "triple": (1000, 2500),
}

_PHASE_THRESHOLDS_2V2 = {
    "early": (7000, 13000),
    "single": (7000, 14000),
    "double": (8000, 15000),
    "triple": (9000, 16000),
}


class BattleStrategy:
    """Manages battle timing and elixir selection strategy.

    Encapsulates the sophisticated elixir selection logic that changes
    based on battle phase, eliminating the need for global variables.
    """

    def __init__(self, fight_mode: str = "Classic 1v1"):
        self.start_time = None
        self.elixir_amounts = [3, 4, 5, 6, 7, 8, 9]

        is_2v2 = fight_mode == "Classic 2v2"
        self.phase_strategies = _PHASE_STRATEGIES
        self.phase_thresholds = _PHASE_THRESHOLDS_2V2 if is_2v2 else _PHASE_THRESHOLDS_1V1

    def start_battle(self):
        """Call when battle begins to start timing."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get seconds elapsed since battle start."""
        return time.time() - self.start_time if self.start_time else 0

    def get_battle_phase(self):
        """Determine current battle phase based on elapsed time."""
        elapsed = self.get_elapsed_time()
        if elapsed < 7:
            return "early"
        elif elapsed < 90:
            return "single"
        elif elapsed < 200:
            return "double"
        else:
            return "triple"

    def select_elixir_amount(self):
        """Select elixir amount to wait for based on current battle phase."""
        phase = self.get_battle_phase()
        weights = self.phase_strategies[phase]
        return random.choices(self.elixir_amounts, weights=weights, k=1)[0]

    def get_thresholds(self):
        """Get (WAIT_THRESHOLD, PLAY_THRESHOLD) for current battle phase."""
        phase = self.get_battle_phase()
        return self.phase_thresholds[phase]


def _fight_loop(
    emulator,
    logger: Logger,
    recording_flag: bool,
    fight_mode: str = "Classic 1v1",
    custom_path: str | None = None,
    observer_mode: bool = False,
) -> bool:
    """Method for handling dynamically timed fight"""
    create_default_bridge_iar(emulator)
    _maybe_start_fight_recording(emulator, logger, recording_flag or observer_mode, fight_mode, custom_path)
    collections.deque(maxlen=3)
    prev_cards_played = logger.get_cards_played()
    battle_detection_lost_count = 0

    # Initialize battle strategy and start timing
    battle_strategy = BattleStrategy(fight_mode)
    battle_strategy.start_battle()
    opponent_tracker.reset()
    recent_card_ids.clear()
    global _hero_ability_available_since
    _hero_ability_available_since = None

    # Initialize online learning trainer & YOLO field detector
    try:
        from pyclashbot.bot.policy import OnlineTrainer, load_torch_model
        from pyclashbot.detection.onnx_detector import load_detector_from_env

        policy_model = load_torch_model(logger=logger, trainable=True)
        setattr(logger, "policy_trainer", OnlineTrainer(model=policy_model, logger=logger))
        setattr(logger, "_field_detector", load_detector_from_env())
    except Exception as exc:
        logger.log(f"Policy / Detector init notice: {exc}")

    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            # If we've lost detection several times in a row, assume the battle
            # ended even if we couldn't confirm it (prevents infinite loops if UI changes).
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        # debug screenshot saving removed from production

        # Get elixir amount and thresholds based on current battle phase
        elixir_amount = battle_strategy.select_elixir_amount()
        wait_threshold, play_threshold = battle_strategy.get_thresholds()

        wait_output = wait_for_elixir(
            emulator,
            logger,
            elixir_amount,
            wait_threshold,
            play_threshold,
            recording_flag,
        )

        if wait_output == "restart":
            logger.change_status("Failed while waiting for elixir")
            return False

        if wait_output == "no battle":
            logger.change_status("Not in battle anymore!")
            break

        if not check_if_in_battle(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status("Battle ended (confirmed)")
                break

            logger.change_status("Lost battle detection — continuing fight loop")
            continue

        play_start_time = time.time()
        if play_a_card(emulator, logger, recording_flag, battle_strategy) is False:
            logger.change_status("Failed to play a card, retrying...")
        # play_time_taken = str(time.time() - play_start_time)[:4]
        logger.change_status(
            f"Made a play in {str(time.time() - play_start_time)[:4]}s",
        )

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Fight complete")
    time.sleep(2.13)
    cards_played = logger.get_cards_played()
    logger.change_status(f"Played ~{cards_played - prev_cards_played} cards this fight")

    return True


def _random_fight_loop(
    emulator,
    logger,
    recording_flag: bool = False,
    fight_mode_chosen: str = "Trophy Road",
    custom_path: str | None = None,
) -> bool:
    """Method for handling dynamically timed fight with random plays"""
    logger.change_status(status="Starting battle with random plays")
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode_chosen, custom_path)
    create_default_bridge_iar(emulator)
    fight_timeout = 5 * 60  # 5 minutes
    start_time = time.time()
    battle_detection_lost_count = 0

    # while in battle:
    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}) — waiting it out",
            )

            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly — assuming battle ended",
                )
                break

            time.sleep(1)
            continue

        battle_detection_lost_count = 0
        if time.time() - start_time > fight_timeout:
            logger.change_status("Random fight loop timed out after 5 minutes")
            return False

        # Clean random-play flow (mirrors rl-bot RandomPlayer): wait for a random
        # elixir gate, then play only if a card is actually available.
        target = random.randint(RANDOM_PLAY_ELIXIR_MIN, RANDOM_PLAY_ELIXIR_MAX)
        elixir_result = wait_for_elixir(emulator, logger, target, recording_flag=recording_flag)
        if elixir_result == "no battle":
            break
        if elixir_result == "restart":
            return False
        play_random_available_card(emulator, logger, recording_flag, time.time() - start_time)

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Random-plays fight complete")
    return True


if __name__ == "__main__":
    pass
