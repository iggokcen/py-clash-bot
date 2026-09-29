"""Live test runner to execute matches on the connected emulator.

Runs live 1v1 fights with the new decision engine, card detection, and YOLO arena vision.
Usage:
    uv run python tools/live_match_test.py [--matches 2] [--mode "Trophy Road"]
"""

from __future__ import annotations

import argparse
import time

from pyclashbot.bot import fight, nav, state_detect
from pyclashbot.emulators.adb import AdbController
from pyclashbot.utils.logger import Logger


def run_live_match(
    emulator: AdbController,
    logger: Logger,
    match_index: int,
    mode: str = "Trophy Road",
) -> bool:
    print(f"\n{'=' * 50}", flush=True)
    print(f"🚀 CANLI TEST - MAÇ #{match_index} BAŞLIYOR ({mode})", flush=True)
    print(f"{'=' * 50}\n", flush=True)

    # If already in battle, finish it first
    if state_detect.check_if_in_battle(emulator):
        print("⚔️ Halihazırda devam eden maç tespit edildi, döngüye bağlanılıyor...", flush=True)
        res = fight._fight_loop(emulator, logger, recording_flag=True, fight_mode=mode)
        fight.end_fight_state(emulator, logger, False, False)
        nav.get_to_main_after_fight(emulator, logger)
        return res

    # Ensure on main menu
    print("📱 Ana menü kontrol ediliyor...", flush=True)
    if not nav.wait_for_clash_main_menu(emulator, logger):
        print("❌ HATA: Clash Royale ana menüsüne ulaşılamadı!", flush=True)
        return False

    print("🟢 Ana menüdeyiz. Maç başlatma butonuna basılıyor...", flush=True)
    time.sleep(1.0)
    if not fight.start_fight(emulator, logger, mode):
        print(f"❌ HATA: start_fight({mode}) başarısız oldu!", flush=True)
        return False

    print("⏳ Rakip aranıyor ve savaşın başlaması bekleniyor...", flush=True)
    if not fight.do_fight_state(
        emulator,
        logger,
        random_fight_mode=False,
        fight_mode_chosen=mode,
        called_from_launching=False,
        recording_flag=True,
    ):
        print("❌ Savaş döngüsü sonlandırıldı veya bir hata oluştu.", flush=True)
        return False

    print("🏁 Savaş bitti! Maç sonu ekranı kapatılıyor...", flush=True)
    fight.end_fight_state(emulator, logger, False, False)
    nav.get_to_main_after_fight(emulator, logger)
    print(f"✅ MAÇ #{match_index} BAŞARIYLA TAMAMLANDI!\n", flush=True)
    return True


class ConsoleLogger(Logger):
    def change_status(self, status: str):
        super().change_status(status)
        print(f"[{time.strftime('%H:%M:%S')}] 🤖 {status}", flush=True)

    def log(self, message: str):
        super().log(message)
        if any(k in message for k in ["Strategy:", "HUD:", "Identified", "Calculated", "Start button", "Made the play"]):
            print(f"[{time.strftime('%H:%M:%S')}] ℹ️ {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Live match test runner")
    parser.add_argument("--matches", type=int, default=2, help="Number of matches to play")
    parser.add_argument("--mode", type=str, default="Trophy Road", help="Fight mode (Trophy Road / Classic 1v1)")
    parser.add_argument("--serial", type=str, default="127.0.0.1:21523", help="ADB serial")
    args = parser.parse_args()

    logger = ConsoleLogger()
    print(f"🔌 Emülatöre bağlanılıyor: {args.serial}...", flush=True)
    emulator = AdbController(logger=logger, device_serial=args.serial)

    total_matches = args.matches
    successful = 0

    for i in range(1, total_matches + 1):
        try:
            ok = run_live_match(emulator, logger, i, mode=args.mode)
            if ok:
                successful += 1
            if i < total_matches:
                print("⏳ Bir sonraki maça geçmeden önce 5 saniye bekleniyor...")
                time.sleep(5.0)
        except KeyboardInterrupt:
            print("\n🛑 Kullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            print(f"\n❌ Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"\n{'=' * 50}")
    print(f"📊 CANLI TEST TAMAMLANDI: {successful}/{total_matches} Maç Başarılı")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
