import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR_MAIN = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "detection", "reference_images", "merge_tactics")

UNITS_DIR = os.path.join(TEMPLATES_DIR_MAIN, "units")
FIELDS_DIR = os.path.join(TEMPLATES_DIR_MAIN, "field")
BENCH_DIR = os.path.join(TEMPLATES_DIR_MAIN, "bench")

GRID_FILE = os.path.join(BASE_DIR, "grid_positions.json")
COMPOSITION_FILE = os.path.join(BASE_DIR, "composition.json")
ELIXIR_FILE = os.path.join(BASE_DIR, "elixir.json")
BENCH_FILE = os.path.join(BASE_DIR, "bench.json")

ADB_PATH = "adb"
THRESHOLD = 0.70

INIT_BATTLE = os.path.join(TEMPLATES_DIR_MAIN, "battle.png")
PHASE_DEPLOY = os.path.join(TEMPLATES_DIR_MAIN, "deploy_phase.png")
PHASE_BATTLE = os.path.join(TEMPLATES_DIR_MAIN, "battle_phase.png")
PLAY_AGAIN_BATTLE = os.path.join(TEMPLATES_DIR_MAIN, "play_again.png")
QUIT = os.path.join(TEMPLATES_DIR_MAIN, "quit.png")
SELL_BUTTON = os.path.join(TEMPLATES_DIR_MAIN, "sell.png")
REWARD = os.path.join(TEMPLATES_DIR_MAIN, "reward.png")
EMPTY_BENCH = os.path.join(TEMPLATES_DIR_MAIN, "empty_bench.png")
BACK = os.path.join(TEMPLATES_DIR_MAIN, "back.png")

def imread_unicode(path):
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        return None

def load_unit_images():
    unit_files = [f for f in os.listdir(UNITS_DIR) if f.endswith(".png")]
    unit_imgs = [imread_unicode(os.path.join(UNITS_DIR, f)) for f in unit_files]
    selected = [False]*len(unit_imgs)
    return unit_files, unit_imgs, selected

def load_field_images():
    field_files = [f for f in os.listdir(FIELDS_DIR) if f.endswith(".png")]
    field_imgs = [imread_unicode(os.path.join(FIELDS_DIR, f)) for f in field_files]
    return field_files, field_imgs

def load_bench_images():
    bench_files = [f for f in os.listdir(BENCH_DIR) if f.endswith(".png")]
    bench_imgs = [imread_unicode(os.path.join(BENCH_DIR, f)) for f in bench_files]
    return bench_files, bench_imgs
