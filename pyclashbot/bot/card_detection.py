import random
import time
from collections import Counter

import numpy

from pyclashbot.bot.coords import CHAMPION_ABILITY_DISMISS_COORD

# Placement profiles: where on screen to tap (left/right), not card role in battle.
# Shared arena zones use zone_* keys; card-named keys stay when only that card (or its evo) uses the region.
# CARD_GROUPS maps card ids → key below (see #674 for full id list including pre-fingerprint evo/hero).
# Visual map: docs/placement-zones.md
PLAY_COORDS = {
    # Troops: bridge line (tanks + heavy win conditions share coords until a follow-up retune).
    "bridge_line": {
        "left": [(115, 332)],
        "right": [(295, 336)],
    },
    # Troops: bridge rush (Hog, Ram, Wall Breakers, …).
    "bridge_rush": {
        "left": [(77, 281), (113, 286), (154, 283)],
        "right": [(257, 283), (300, 284), (353, 283)],
    },
    # Troops/buildings: back field support (swarms, spawners, ranged support).
    "back_support": {
        "left": [(69, 442), (158, 444), (166, 394), (102, 451)],
        "right": [(247, 396), (264, 440), (343, 442), (312, 456)],
    },
    # Troops: king-tower lane (Witch, Archer Queen, …).
    "king_lane": {
        "left": [(70, 463), (184, 398), (166, 394), (191, 473)],
        "right": [(247, 396), (264, 440), (343, 463), (211, 471)],
    },
    # Princess (+ evo): deep lane, tuned separately from king_lane.
    "princess": {
        "left": [(70, 463), (240, 400), (191, 471), (220, 402)],
        "right": [(343, 463), (184, 398), (211, 473), (166, 394)],
    },
    # Buildings: defensive structures in the back field.
    "defense_building": {
        "left": [(224, 320), (191, 351), (182, 362), (202, 339)],
        "right": [(224, 320), (224, 334), (214, 360), (211, 381)],
    },
    # Buildings: siege (Mortar, X-Bow).
    "siege_building": {
        "left": [(122, 311), (125, 312)],
        "right": [(292, 311), (298, 312)],
    },
    # Spells: center field behind bridge (Mirror, Barb Barrel, Royal Delivery).
    "center_spell": {
        "left": [(116, 160)],
        "right": [(302, 160)],
    },
    "fireball": {
        "left": [(116, 134)],
        "right": [(302, 134)],
    },
    # Spells: standard lane target (Fireball, Poison, Log, …).
    "lane_spell": {
        "left": [(118, 185)],
        "right": [(295, 185)],
    },
    # Ground roll / friendly-only spells (Log, Barb Barrel, Royal Delivery)
    # MUST be placed on friendly side of the river (y >= 283) rolling across the bridge.
    "ground_spell": {
        "left": [(118, 305), (118, 325)],
        "right": [(295, 305), (295, 325)],
    },
    "large_spell": {"left": [(170, 196), (168, 194)], "right": [(249, 196), (248, 193)]},
    # Spells: small reactive (Zap, Void, Clone, Rage, Goblin Curse, …).
    "reactive_spell": {
        "left": [(118, 185)],
        "right": [(295, 185)],
    },
    # Troops: 1-elixir spirits (played on your side).
    # Same coords as defense_building until spirit lane is tuned.
    "spirit": {
        "left": [(224, 320), (191, 351), (182, 362), (202, 339)],
        "right": [(224, 320), (224, 334), (214, 360), (211, 381)],
    },
    "tornado": {
        "left": [(50, 182)],
        "right": [(355, 178)],
    },
    "rocket": {
        "left": [(127, 161)],
        "right": [(282, 162)],
    },
    "goblin_barrel": {
        "left": [(115, 161), (116, 161), (117, 161)],
        "right": [(300, 161), (302, 161), (301, 161)],
    },
    "graveyard": {
        "left": [(88, 157)],
        "right": [(325, 156)],
    },
    "miner": {
        "left": [(86, 156), (90, 104), (143, 113), (142, 153)],
        "right": [(274, 152), (276, 111), (339, 111), (323, 157)],
    },
    "goblin_drill": {
        "left": [(91, 149), (122, 183)],
        "right": [(305, 177), (334, 154)],
    },
}


# Maps card ids → arena zone profile → PLAY_COORDS (see #674).
# Includes base, Evolution, and Hero ids even before card_color_data fingerprints exist so
# new detection work automatically picks up tuned clicks instead of the No group fallback.
CARD_GROUPS: dict[str, list[str]] = {
    "king_lane": [
        "archer_queen",
        "electro_dragon",
        "evo_electro_dragon",
        "goblin_demolisher",
        "little_prince",
        "mega_minion",
        "hero_mega_minion",
        "mother_witch",
        "night_witch",
        "royal_recruits",
        "evo_royal_recruits",
        "three_musketeers",
        "witch",
        "evo_witch",
    ],
    "princess": [
        "princess",
        "evo_princess",
    ],
    "bridge_line": [
        "balloon",
        "hero_balloon",
        "electro_giant",
        "elixir_golem",
        "giant",
        "hero_giant",
        "giant_skeleton",
        "goblin_giant",
        "evo_goblin_giant",
        "golem",
        "lava_hound",
        "royal_giant",
        "evo_royal_giant",
        "rune_giant",
        "sparky",
        "bandit",
        "barbarians",
        "evo_barbarians",
        "boss_bandit",
        "bowler",
        "hero_bowler",
        "dark_prince",
        "hero_dark_prince",
        "elite_barbarians",
        "fisherman",
        "golden_knight",
        "guards",
        "ice_golem",
        "hero_ice_golem",
        "knight",
        "evo_knight",
        "hero_knight",
        "mega_knight",
        "evo_mega_knight",
        "mighty_miner",
        "mini_pekka",
        "hero_mini_pekka",
        "monk",
        "pekka",
        "evo_pekka",
        "clone",
        "prince",
        "royal_ghost",
        "evo_royal_ghost",
        "skeleton_king",
        "valkyrie",
        "evo_valkyrie",
        "berserker",
        "spirit_empress",
    ],
    "center_spell": [
        "mirror",
    ],
    "ground_spell": [
        "barb_barrel",
        "hero_barb_barrel",
        "log",
        "the_log",
        "royal_delivery",
    ],
    "fireball": [
        "fireball",
    ],
    "lane_spell": [
        "snowball",
        "evo_snowball",
    ],
    "large_spell": ["earthquake", "arrows", "freeze", "lightning", "poison"],
    "spirit": [
        "electro_spirit",
        "fire_spirit",
        "heal_spirit",
        "ice_spirit",
        "evo_ice_spirit",
    ],
    "reactive_spell": [
        "gob_curse",
        "rage",
        "void",
        "vines",
        "zap",
        "evo_zap",
    ],
    "rocket": ["rocket"],
    "tornado": ["tornado"],
    "goblin_drill": ["goblin_drill", "evo_goblin_drill"],
    "graveyard": ["graveyard"],
    "defense_building": [
        "bomb_tower",
        "cannon",
        "evo_cannon",
        "cannon_cart",
        "goblin_cage",
        "evo_goblin_cage",
        "goblinstein",
        "inferno_tower",
        "tesla",
        "evo_tesla",
    ],
    "bridge_rush": [
        "battle_ram",
        "evo_battle_ram",
        "hog",
        "lumberjack",
        "evo_lumberjack",
        "ram_rider",
        "royal_hogs",
        "evo_royal_hogs",
        "skeleton_barrel",
        "evo_skeleton_barrel",
        "suspicious_bush",
        "wall_breakers",
        "evo_wall_breakers",
    ],
    "miner": [
        "miner",
    ],
    "goblin_barrel": [
        "goblin_barrel",
        "evo_goblin_barrel",
    ],
    "siege_building": [
        "mortar",
        "evo_mortar",
        "xbow",
    ],
    "back_support": [
        "archers",
        "evo_archers",
        "baby_dragon",
        "evo_baby_dragon",
        "barb_hut",
        "bats",
        "evo_bats",
        "battle_healer",
        "bomber",
        "evo_bomber",
        "dart_goblin",
        "evo_dart_goblin",
        "electro_wizard",
        "elixir_collector",
        "executioner",
        "evo_executioner",
        "fire_cracker",
        "evo_fire_cracker",
        "flying_machine",
        "furnace",
        "evo_furnace",
        "goblin_gang",
        "goblin_hut",
        "goblin_machine",
        "goblins",
        "hero_goblins",
        "hunter",
        "evo_hunter",
        "ice_wizard",
        "inferno_dragon",
        "evo_inferno_dragon",
        "magic_archer",
        "hero_magic_archer",
        "minion_horde",
        "evo_minion_horde",
        "minions",
        "musketeer",
        "evo_musketeer",
        "hero_musketeer",
        "phoenix",
        "rascals",
        "skeleton_army",
        "evo_skeleton_army",
        "skeleton_dragons",
        "skeletons",
        "evo_skeletons",
        "spear_goblins",
        "tombstone",
        "hero_tombstone",
        "wizard",
        "evo_wizard",
        "hero_wizard",
        "zappies",
    ],
}

# card classification data used for potential future filtering/analytics; currently informational only
CARD_ATTRIBUTES: dict[str, dict[str, str]] = {
    "cannon": {"type": "Building", "subtype": "Ground defense"},
    "cannon_cart": {"type": "Troop/Building", "subtype": "Mobile defense"},
    "tesla": {"type": "Building", "subtype": "Mixed defense"},
    "bomb_tower": {"type": "Building", "subtype": "Anti-swarms"},
    "inferno_tower": {"type": "Building", "subtype": "Anti-tank"},
    "mortar": {"type": "Building", "subtype": "Win condition / long-range"},
    "xbow": {"type": "Building", "subtype": "Win condition / cycle"},
    "tombstone": {"type": "Building", "subtype": "Spawner / defense"},
    "elixir_collector": {"type": "Building", "subtype": "Economy / support"},
    "barb_hut": {"type": "Building", "subtype": "Spawner defense"},
    "goblin_hut": {"type": "Building", "subtype": "Spawner / pressure"},
    "furnace": {"type": "Building", "subtype": "Chip pressure"},
    "goblin_cage": {"type": "Building", "subtype": "Tank-lure defense"},
    "knight": {"type": "Troop", "subtype": "Mini-tank / defense"},
    "barbarians": {"type": "Troop", "subtype": "Ground defense"},
    "elite_barbarians": {"type": "Troop", "subtype": "Pressure attack"},
    "giant": {"type": "Troop", "subtype": "Tank / Win condition"},
    "royal_giant": {"type": "Troop", "subtype": "Win condition"},
    "golem": {"type": "Troop", "subtype": "Tank / Win condition"},
    "elixir_golem": {"type": "Troop", "subtype": "Offensive tank"},
    "giant_skeleton": {"type": "Troop", "subtype": "Anti-push"},
    "mini_pekka": {"type": "Troop", "subtype": "Anti-tank"},
    "pekka": {"type": "Troop", "subtype": "Heavy anti-tank"},
    "prince": {"type": "Troop", "subtype": "Tower attacker"},
    "dark_prince": {"type": "Troop", "subtype": "Ground splash"},
    "golden_knight": {"type": "Champion", "subtype": "Assassin / pressure"},
    "monk": {"type": "Champion", "subtype": "Control / reflect"},
    "mighty_miner": {"type": "Champion", "subtype": "Mobile win pressure"},
    "bandit": {"type": "Troop", "subtype": "Dash assassin"},
    "royal_ghost": {"type": "Troop", "subtype": "Stealth control"},
    "valkyrie": {"type": "Troop", "subtype": "Splash defense"},
    "witch": {"type": "Troop", "subtype": "Skeleton support"},
    "night_witch": {"type": "Troop", "subtype": "Bat support"},
    "hunter": {"type": "Troop", "subtype": "Close-range burst"},
    "bomber": {"type": "Troop", "subtype": "Ground splash"},
    "spear_goblins": {"type": "Troop", "subtype": "Cycle poke"},
    "goblins": {"type": "Troop", "subtype": "Cycle defense"},
    "goblin_gang": {"type": "Troop", "subtype": "Fast defense"},
    "skeletons": {"type": "Troop", "subtype": "Cycle"},
    "skeleton_army": {"type": "Troop", "subtype": "Mass ground defense"},
    "battle_ram": {"type": "Troop", "subtype": "Win condition"},
    "hog": {"type": "Troop", "subtype": "Direct win condition"},
    "ram_rider": {"type": "Troop", "subtype": "Win condition hybrid"},
    "goblin_giant": {"type": "Troop", "subtype": "Offensive tank"},
    "dart_goblin": {"type": "Troop", "subtype": "Chip long-range"},
    "ice_golem": {"type": "Troop", "subtype": "Cheap tank / slow"},
    "balloon": {"type": "Troop", "subtype": "Win condition / tower damage"},
    "lava_hound": {"type": "Troop", "subtype": "Air tank / win condition"},
    "baby_dragon": {"type": "Troop", "subtype": "Air splash"},
    "inferno_dragon": {"type": "Troop", "subtype": "Air anti-tank"},
    "skeleton_dragons": {"type": "Troop", "subtype": "Support air"},
    "minions": {"type": "Troop", "subtype": "Air defense"},
    "minion_horde": {"type": "Troop", "subtype": "High DPS air"},
    "flying_machine": {"type": "Troop", "subtype": "Long-range support"},
    "phoenix": {"type": "Troop", "subtype": "Defensive/Counter push"},
    "wizard": {"type": "Troop", "subtype": "Splash support"},
    "ice_wizard": {"type": "Troop", "subtype": "Slow control"},
    "electro_wizard": {"type": "Troop", "subtype": "Stun & reset"},
    "magic_archer": {"type": "Troop", "subtype": "Long-line chip"},
    "archers": {"type": "Troop", "subtype": "Cheap ranged defense"},
    "battle_healer": {"type": "Troop", "subtype": "Healing support"},
    "graveyard": {"type": "Spell", "subtype": "Win condition"},
    "miner": {"type": "Troop", "subtype": "Chip win condition"},
    "sparky": {"type": "Troop", "subtype": "Massive tower damage"},
    "arrows": {"type": "Spell", "subtype": "Anti-swarms"},
    "snowball": {"type": "Spell", "subtype": "Control knockback"},
    "zap": {"type": "Spell", "subtype": "Reset & cleanup"},
    "log": {"type": "Spell", "subtype": "Ground chip + knockback"},
    "barb_barrel": {"type": "Spell", "subtype": "Ground control"},
    "fireball": {"type": "Spell", "subtype": "Medium damage"},
    "poison": {"type": "Spell", "subtype": "Zone control"},
    "lightning": {"type": "Spell", "subtype": "Support removal"},
    "rocket": {"type": "Spell", "subtype": "High damage"},
    "rage": {"type": "Spell", "subtype": "Offensive buff"},
    "heal_spirit": {"type": "Spell", "subtype": "Healing burst"},
    "freeze": {"type": "Spell", "subtype": "Full control"},
    "tornado": {"type": "Spell", "subtype": "Group control"},
    "mirror": {"type": "Spell", "subtype": "Duplicate card"},
    "clone": {"type": "Spell", "subtype": "Offensive synergy"},
    "royal_delivery": {"type": "Spell", "subtype": "Defensive burst"},
    "goblin_barrel": {"type": "Spell", "subtype": "Chip win condition"},
    "skeleton_barrel": {"type": "Spell", "subtype": "Surprise pressure"},
}

toplefts = [
    (115, 529),
    (182, 529),
    (249, 529),
    (316, 529),
]

TOTAL_WIDTH = 54
TOTAL_HEIGHT = 66
HALF_WIDTH = int(TOTAL_WIDTH / 2)
HALF_HEIGHT = int(TOTAL_HEIGHT / 2)

# color stuff
CARD_MATCH_THRESHOLD = 1000

# Spirit Empress hand art switches between ground/air by elixir; match either variant.
CARD_DETECTION_ALIASES: dict[str, str] = {
    "spirit_empress_air": "spirit_empress",
    "spirit_empress_ground": "spirit_empress",
}

COLORS = {
    "Red": [255, 0, 0],
    "Orange": [255, 165, 0],
    "Yellow": [255, 255, 0],
    "Green": [0, 128, 0],
    "Blue": [0, 0, 255],
    "Indigo": [75, 0, 130],
    "Violet": [148, 0, 211],
    "Cyan": [0, 255, 255],
    "Magenta": [255, 0, 255],
    "Pink": [255, 192, 203],
    "Turquoise": [64, 224, 208],
    "Lime": [0, 255, 0],
    "Purple": [128, 0, 128],
    "Brown": [165, 42, 42],
    "Teal": [0, 128, 128],
    "Maroon": [128, 0, 0],
}

COLORS_ARRAY = numpy.array(list(COLORS.values()))
COLORS_KEYS = list(COLORS.keys())



_FINGERPRINTS_DB: dict[str, object] | None = None


def get_all_card_fingerprints() -> dict[str, object]:
    """Retrieve the universal 123-card visual fingerprint database."""
    global _FINGERPRINTS_DB
    if _FINGERPRINTS_DB is None:
        from pathlib import Path
        p = Path(__file__).resolve().parent / "card_fingerprints.json"
        if p.is_file():
            import json
            try:
                _FINGERPRINTS_DB = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                _FINGERPRINTS_DB = {}
        else:
            _FINGERPRINTS_DB = {}
    return _FINGERPRINTS_DB

_card_color_data_transformed: dict[str, list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]]]] = {}
for card_name, card_data in get_all_card_fingerprints().items():
    if card_data and isinstance(card_data, list) and isinstance(card_data[0], dict):
        _card_color_data_transformed[card_name] = [numpy.array(list(corner.values())) for corner in card_data]
    else:
        # Already transformed
        _card_color_data_transformed[card_name] = card_data
card_color_data: dict[str, list[numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]]]] = (
    _card_color_data_transformed
)

# card color data (initially holds dict[str, int] entries, then transformed to ndarray)




def calculate_offset(card_name, card_data, collected_data_array):
    total_offset = 0
    for i, corner_data_array in enumerate(card_data):
        offset = numpy.sum(numpy.abs(collected_data_array[i] - corner_data_array))
        total_offset += offset
    return card_name, total_offset




ACTIVE_DECK_CARD_IDS: list[str] = []


def set_active_deck_cards(card_ids: list[str]) -> None:
    """Set the known 8 card IDs from the player's active deck (e.g. from CR API)."""
    global ACTIVE_DECK_CARD_IDS
    ACTIVE_DECK_CARD_IDS = [c.lower().strip() for c in card_ids]





def get_active_deck_cards() -> list[str]:
    return list(ACTIVE_DECK_CARD_IDS)


def find_closest_card(collected_data):
    best_card = None
    best_offset = CARD_MATCH_THRESHOLD + 1
    # Debug offsets disabled in production
    debug_offsets = None

    collected_data_array = numpy.array(
        [list(corner.values()) for corner in collected_data],
    )

    # If active deck is known (from Clash Royale API), prioritize matching against those 8 cards
    if ACTIVE_DECK_CARD_IDS:
        deck_best_card = None
        deck_best_offset = CARD_MATCH_THRESHOLD + 1
        for card_name in ACTIVE_DECK_CARD_IDS:
            if card_name in card_color_data:
                _, total_offset = calculate_offset(
                    card_name,
                    card_color_data[card_name],
                    collected_data_array,
                )
                if total_offset < deck_best_offset:
                    deck_best_offset = total_offset
                    deck_best_card = card_name
        if deck_best_card is not None and deck_best_offset <= CARD_MATCH_THRESHOLD:
            return deck_best_card

    for card_name, card_data in card_color_data.items():
        card_name, total_offset = calculate_offset(  # noqa: PLW2901
            card_name,
            card_data,
            collected_data_array,
        )
        if total_offset < best_offset:
            best_offset = total_offset
            best_card = card_name
        if debug_offsets is not None:
            debug_offsets.append((card_name, total_offset))

    # Debugging output removed from production.

    if best_offset > CARD_MATCH_THRESHOLD:
        return "UNKNOWN"

    return best_card


# identification methods
def make_pixel_dict_from_color_list(color_list):
    # Initialize the pixel_dict with zeros
    pixel_dict = dict.fromkeys(COLORS.keys(), 0)

    # Count the occurrences of each color in color_list
    color_counts = Counter(color_list)

    # Update the pixel_dict with the counts
    pixel_dict.update(color_counts)

    return pixel_dict


def color_from_pixel(pixels):
    # Calculate the Euclidean distance between the pixels and each color
    distances = numpy.linalg.norm(COLORS_ARRAY - pixels[:, None], axis=2)

    # Find the color with the minimum distance for each pixel
    closest_colors = [COLORS_KEYS[i] for i in numpy.argmin(distances, axis=1)]

    return closest_colors


def get_corner_pixels(x_range, y_range, iar):
    # Get all pixels in the range
    pixels = iar[y_range[0] : y_range[1], x_range[0] : x_range[1]].reshape(-1, 3)

    # Get the color for each pixel
    colors = color_from_pixel(pixels)

    return make_pixel_dict_from_color_list(colors)


def get_all_pixel_data(emulator, chosen_card_index):
    topleft = toplefts[chosen_card_index]

    corners = [
        ((topleft[0], topleft[0] + HALF_WIDTH), (topleft[1], topleft[1] + HALF_HEIGHT)),
        (
            (topleft[0] + HALF_WIDTH, topleft[0] + TOTAL_WIDTH),
            (topleft[1], topleft[1] + HALF_HEIGHT),
        ),
        (
            (topleft[0], topleft[0] + HALF_WIDTH),
            (topleft[1] + HALF_HEIGHT, topleft[1] + TOTAL_HEIGHT),
        ),
        (
            (topleft[0] + HALF_WIDTH, topleft[0] + TOTAL_WIDTH),
            (topleft[1] + HALF_HEIGHT, topleft[1] + TOTAL_HEIGHT),
        ),
    ]

    color_list = [get_corner_pixels(*corner, battle_iar) for corner in corners]

    # print(f"card_name: {color_list},")
    return color_list


purple_color = numpy.array([255, 43, 227])
card_toplefts = numpy.array(
    [
        [133, 582],
        [199, 583],
        [266, 583],
        [334, 582],
    ],
)

# Pre-calculate x_coords and y_coords for each card
card_coords = [
    (
        numpy.arange(topleft[0], topleft[0] + 20),
        numpy.arange(topleft[1], topleft[1] + 20),
    )
    for topleft in card_toplefts
]

play_side = "left"
battle_iar: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]] | None = None


def is_hero_champion_ability_visible(emulator) -> bool:
    iar = emulator.screenshot()
    pixels = numpy.array([iar[462][324], iar[453][334], iar[462][336]])
    colors = numpy.array(
        [
            [215, 28, 223],
            [240, 39, 254],
            [239, 40, 251],
        ],
    )

    for p in pixels:
        if numpy.any(numpy.all(numpy.abs(colors - p) <= 30, axis=1)):
            return True

    return False


def check_which_cards_are_available(emulator, check_side=False):
    global battle_iar
    battle_iar = emulator.screenshot()
    card_exists_list = []

    if check_side:
        global play_side
        _, play_side = switch_side()

    for i, coords in enumerate(card_coords):
        x_coords, y_coords = coords
        iar_pixels = battle_iar[numpy.ix_(y_coords, x_coords)]
        purple_pixels = numpy.all(numpy.abs(iar_pixels - purple_color) <= 30, axis=-1)
        count = numpy.sum(purple_pixels)
        if count >= 26:
            card_exists_list.append(i)

    return card_exists_list


def trigger_hero_champion_ability(emulator, logger) -> None:
    emulator.click(*CHAMPION_ABILITY_DISMISS_COORD)
    logger.change_status("Triggered Hero/Champion ability")


_hand_classifier = None


def _get_hand_classifier():
    global _hand_classifier
    if _hand_classifier is None:
        try:
            from pyclashbot.detection.hand_classifier import load_hand_classifier

            _hand_classifier = load_hand_classifier()
        except Exception:
            _hand_classifier = False
    return _hand_classifier if _hand_classifier is not False else None


def _classify_card_slot(iar: numpy.ndarray, card_index: int) -> str | None:
    clf = _get_hand_classifier()
    if clf is None or iar is None or iar.size == 0:
        return None
    try:
        from pyclashbot.bot.coords import HAND_CARDS_COORDS

        if 0 <= card_index < len(HAND_CARDS_COORDS):
            cx, cy = HAND_CARDS_COORDS[card_index]
            y1 = max(0, cy - 35)
            y2 = min(iar.shape[0], cy + 35)
            x1 = max(0, cx - 30)
            x2 = min(iar.shape[1], cx + 30)
            crop = iar[y1:y2, x1:x2]
            if crop.size > 0:
                card_name, conf = clf.predict(crop)
                if conf >= 0.55:
                    return CARD_DETECTION_ALIASES.get(card_name, card_name)
    except Exception:
        pass
    return None


def identify_hand_cards(emulator, card_index):
    iar = emulator.screenshot() if emulator is not None else battle_iar
    if iar is not None:
        cnn_pred = _classify_card_slot(iar, card_index)
        if cnn_pred:
            return cnn_pred
    color_chosen_card = get_all_pixel_data(emulator, card_index)
    card_id = find_closest_card(color_chosen_card)
    return CARD_DETECTION_ALIASES.get(card_id, card_id)


def identify_hand_cards_from_frame(card_index):
    """Identify a hand slot from the already-captured ``battle_iar``."""
    if battle_iar is not None:
        cnn_pred = _classify_card_slot(battle_iar, card_index)
        if cnn_pred:
            return cnn_pred
    card_id = find_closest_card(get_all_pixel_data(None, card_index))
    return CARD_DETECTION_ALIASES.get(card_id, card_id)


def identify_available_hand(indices):
    """Map hand slot index -> card id for every playable slot in the current frame."""
    hand: dict[int, str] = {}
    for index in indices:
        try:
            hand[index] = identify_hand_cards_from_frame(index)
        except (AssertionError, ValueError, TypeError):
            # A slot that cannot be fingerprinted is simply not offered to the
            # strategy engine; it is not an error, just an unknown card.
            continue
    return hand


# Create the reverse lookup dictionary
CARD_TO_GROUP = {card: group for group, cards in CARD_GROUPS.items() for card in cards}


def get_card_group(card_id) -> str:
    # Use the reverse lookup dictionary for O(1) lookups
    return CARD_TO_GROUP.get(card_id, "No group")


def get_play_coords_for_card(emulator, logger, card_index, elapsed_time: float = 0):
    # get the ID of this card(ram_rider, zap, etc)
    id_cards_start_time = time.time()
    identity = identify_hand_cards(emulator, card_index)
    time_taken = str(time.time() - id_cards_start_time)[:3]
    logger.change_status(f"Identified card as {identity} ({time_taken}s)")

    # get the grouping of this card (hog, turret, spell, etc)
    group = get_card_group(identity)

    # get the play coords of this grouping
    coords = calculate_play_coords(group, play_side, elapsed_time)

    return identity, coords


def calculate_play_coords(card_grouping: str, side_preference: str, elapsed_time: float = 0):
    # if there is a dedicated coordinate for this card
    if card_grouping == "No group":
        # Active bridge attack: deploy directly at the bridge so troops cross and pressure enemy tower
        if side_preference == "left":
            return (random.randint(90, 150), random.randint(281, 315))
        return (random.randint(270, 330), random.randint(281, 315))

    if PLAY_COORDS.get(card_grouping):
        group_datum = PLAY_COORDS[card_grouping]
        if side_preference == "left" and "left" in group_datum:
            return random.choice(group_datum["left"])
        if side_preference == "right" and "right" in group_datum:
            return random.choice(group_datum["right"])
        if "coords" in group_datum:
            return random.choice(group_datum["coords"])

    # Fallback safe friendly spawn zone if grouping has no matching side coords
    if side_preference == "left":
        return (random.randint(60, 206), random.randint(380, 480))
    return (random.randint(210, 351), random.randint(380, 480))


bridge_iar: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]] | None = None


def create_default_bridge_iar(emulator):
    global bridge_iar
    bridge_iar = emulator.screenshot()


bridge_pixel = [[100, 200], [275, 200]]


def switch_side():
    assert battle_iar is not None, "battle_iar must be set before calling switch_side()"
    assert bridge_iar is not None, "bridge_iar must be set before calling switch_side()"
    bridge_color_offset = []
    for i, bridge in enumerate(bridge_pixel):
        all_coords = [(y, x) for x in range(bridge[0], bridge[0] + 40) for y in range(bridge[1], bridge[1] + 175)]
        pixel_coords = numpy.array(all_coords)
        iar_pixels = battle_iar[pixel_coords[:, 0], pixel_coords[:, 1]]
        bridge_iar_pixels = bridge_iar[pixel_coords[:, 0], pixel_coords[:, 1]]
        bridge_color_offset.append(numpy.linalg.norm(iar_pixels - bridge_iar_pixels))

    if bridge_color_offset[0] > bridge_color_offset[1]:
        return bridge_color_offset[0], "left"
    return bridge_color_offset[1], "right"


if __name__ == "__main__":
    all_data = get_all_pixel_data(12, 0)
    for data in all_data:
        id = find_closest_card(data)
        if id == "UNKNOWN":
            print(data)
        print(id)
