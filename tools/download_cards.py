import time
import urllib.request
from pathlib import Path

from pyclashbot.utils.cr_api import ClashRoyaleClient, normalize_card_name

out_dir = Path("pyclashbot/detection/reference_images/cards")
out_dir.mkdir(parents=True, exist_ok=True)

client = ClashRoyaleClient()
cards = client.get_cards()
print(f"Total cards from API: {len(cards)}")

saved = 0
failed = []
for c in cards:
    name_id = normalize_card_name(c["name"])
    icon_url = c.get("iconUrls", {}).get("medium")
    if not icon_url:
        continue
    file_path = out_dir / f"{name_id}.png"
    if not file_path.exists():
        try:
            req = urllib.request.Request(icon_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                file_path.write_bytes(resp.read())
            saved += 1
            time.sleep(0.05)
        except Exception as e:
            failed.append((name_id, str(e)))

total_files = len(list(out_dir.glob("*.png")))
print(f"Finished! New downloads: {saved}, Total in folder: {total_files}, Failed: {len(failed)}")
if failed:
    print("Failed cards:", failed[:5])
