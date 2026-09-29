"""Live Match Observer: monitors live battles, captures recordings, and logs AI decisions."""
from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

from pyclashbot.bot.fight import do_fight_state, end_fight_state, start_fight
from pyclashbot.bot.nav import wait_for_clash_main_menu
from pyclashbot.emulators.adb import AdbController
from pyclashbot.utils.caching import USER_SETTINGS_CACHE
from pyclashbot.utils.logger import Logger

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable


class CallbackLogger(Logger):
    """Logger that mirrors events to an optional GUI callback and the console."""

    def __init__(self, callback: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.callback = callback

    def _emit(self, text: str) -> None:
        if self.callback is not None:
            try:
                self.callback(text)
            except Exception:
                pass

    def change_status(self, status: str, *args, **kwargs) -> None:
        super().change_status(status, *args, **kwargs)
        msg = f"[STATUS] {status}"
        print(msg, flush=True)
        self._emit(msg)

    def log(self, message: str, *args, **kwargs) -> None:
        super().log(message, *args, **kwargs)
        if any(k in message for k in ["Strategy:", "HUD:", "Clicked", "Error", "Failed", "[+]"]):
            msg = f"[LOG] {message}"
            print(msg, flush=True)
            self._emit(msg)


def resolve_device_serial(preferred: str | None = None) -> str:
    """Find the best ADB serial from arguments, active devices, or cache."""
    if preferred and preferred.strip():
        return preferred.strip()

    try:
        devices = AdbController.detect_devices()
        if devices:
            return devices[0]
    except Exception:
        pass

    if USER_SETTINGS_CACHE.exists():
        try:
            cached = USER_SETTINGS_CACHE.load_data()
            for key in ["gp_device_serial", "bs_device_serial", "adb_serial"]:
                val = cached.get(key)
                if val and str(val).strip():
                    return str(val).strip()
        except Exception:
            pass

    return "127.0.0.1:21523"


def run_live_match(
    match_number: int,
    total_matches: int,
    serial: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> dict | None:
    """Run a single live match with recording and observer metrics."""
    target_serial = resolve_device_serial(serial)
    logger = CallbackLogger(callback=log_callback)

    header = f"\n{'='*55}\n>>> LIVE MATCH OBSERVER: MATCH {match_number}/{total_matches}\n>>> DEVICE: {target_serial}\n{'='*55}"
    print(header, flush=True)
    if log_callback:
        log_callback(header)

    try:
        emu = AdbController(logger=logger, device_serial=target_serial)
    except Exception as e:
        err = f"[!] Failed to connect to emulator ({target_serial}): {e}"
        print(err, flush=True)
        if log_callback:
            log_callback(err)
        return None

    if stop_event and stop_event.is_set():
        return None

    if not wait_for_clash_main_menu(emu, logger):
        msg = f"[!] Match {match_number}: Not on main menu at start!"
        print(msg, flush=True)
        if log_callback:
            log_callback(msg)
        return None

    mode = "Classic 1v1"
    start_msg = f"[+] On main menu. Starting {mode} battle..."
    print(start_msg, flush=True)
    if log_callback:
        log_callback(start_msg)

    if not start_fight(emu, logger, mode):
        fail_msg = f"[!] Match {match_number}: start_fight failed!"
        print(fail_msg, flush=True)
        if log_callback:
            log_callback(fail_msg)
        return None

    start_time = time.time()
    fight_ok = do_fight_state(
        emulator=emu,
        logger=logger,
        random_fight_mode=False,
        fight_mode_chosen=mode,
        called_from_launching=False,
        recording_flag=True,
    )
    duration = time.time() - start_time

    end_ok = end_fight_state(emu, logger, recording_flag=True, disable_win_tracker_toggle=False)

    cards_played = getattr(logger, "cards_played", 0)
    wins = getattr(logger, "wins", 0)
    losses = getattr(logger, "losses", 0)

    summary_msg = (
        f"[+] Match {match_number} Summary:\n"
        f"    Duration: {duration:.1f}s | Cards Played: {cards_played}\n"
        f"    Wins: {wins} | Losses: {losses} | Fight OK: {fight_ok} | End OK: {end_ok}"
    )
    print(summary_msg, flush=True)
    if log_callback:
        log_callback(summary_msg)

    return {
        "match": match_number,
        "duration": duration,
        "cards_played": cards_played,
        "win": wins > 0,
        "end_ok": end_ok,
    }


def start_observer_session(
    total_matches: int = 1,
    serial: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> list[dict]:
    """Run an observer session across multiple matches, supporting clean interruption."""
    results: list[dict] = []
    for i in range(1, total_matches + 1):
        if stop_event and stop_event.is_set():
            msg = "[!] Observer session stopped by user."
            print(msg, flush=True)
            if log_callback:
                log_callback(msg)
            break

        res = run_live_match(
            match_number=i,
            total_matches=total_matches,
            serial=serial,
            log_callback=log_callback,
            stop_event=stop_event,
        )
        if res:
            results.append(res)

        if i < total_matches:
            for _ in range(5):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(1)

    complete_msg = f"\n{'='*55}\nOBSERVER SESSION COMPLETED: {len(results)}/{total_matches} matches recorded.\n{'='*55}"
    print(complete_msg, flush=True)
    if log_callback:
        log_callback(complete_msg)

    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dev = sys.argv[2] if len(sys.argv) > 2 else None
    start_observer_session(total_matches=n, serial=dev)
