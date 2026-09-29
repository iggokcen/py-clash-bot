import json
from pathlib import Path
import cv2
import numpy as np

CARDS_DIR = Path("pyclashbot/detection/reference_images/cards")
OUT_FILE = Path("pyclashbot/bot/card_fingerprints.json")


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image."""
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = resized[:, 1:] > resized[:, :-1]
    return "".join(f"{val:x}" for val in np.packbits(diff.flatten()))


def compute_color_vector(img: np.ndarray) -> list[float]:
    """Compute a compact 32-bin HSV color histogram."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()


def build_database():
    database = {}
    for img_path in sorted(CARDS_DIR.glob("*.png")):
        card_id = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Crop center artwork (skip outer border)
        center = img[int(h * 0.15) : int(h * 0.85), int(w * 0.1) : int(w * 0.9)]
        dhash = compute_dhash(center)
        hist = compute_color_vector(center)
        database[card_id] = {
            "dhash": dhash,
            "color_hist": hist,
        }

    OUT_FILE.write_text(json.dumps(database, indent=2), encoding="utf-8")
    print(f"Built card fingerprints database for {len(database)} cards -> {OUT_FILE}")


if __name__ == "__main__":
    build_database()
