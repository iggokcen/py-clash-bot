import urllib.request
import json
import ssl
from typing import List

# Mapping from API card names to pyclashbot card IDs
# You might need to add or tweak mappings based on EXACT name matches.
# E.g., API: "Mini P.E.K.K.A" -> pyclashbot: "mini_pekka"
API_TO_CARD_ID_MAP = {
    "Mini P.E.K.K.A": "mini_pekka",
    "P.E.K.K.A": "pekka",
    "X-Bow": "xbow",
    "Elixir Collector": "elixir_collector",
    "Royal Hogs": "royal_hogs",
    "Elite Barbarians": "elite_barbarians",
    "Goblin Barrel": "goblin_barrel",
    "Goblin Gang": "goblin_gang",
    "Goblin Giant": "goblin_giant",
    "Ice Spirit": "ice_spirit",
    "Ice Golem": "ice_golem",
    "Ice Wizard": "ice_wizard",
    "Electro Wizard": "electro_wizard",
    "Electro Dragon": "electro_dragon",
    "Electro Giant": "electro_giant",
    "Electro Spirit": "electro_spirit",
    "Magic Archer": "magic_archer",
    "Royal Giant": "royal_giant",
    "Royal Ghost": "royal_ghost",
    "Royal Delivery": "royal_delivery",
    "Royal Recruits": "royal_recruits",
    "Mega Knight": "mega_knight",
    "Mega Minion": "mega_minion",
    "Dart Goblin": "dart_goblin",
    "Spear Goblins": "spear_goblins",
    "Skeleton Barrel": "skeleton_barrel",
    "Skeleton Army": "skeleton_army",
    "Skeleton Dragons": "skeleton_dragons",
    "Skeleton King": "skeleton_king",
    "Tombstone": "tombstone",
    "Barbarian Barrel": "barb_barrel",
    "Barbarian Hut": "barbarian_hut",
    "Goblin Hut": "goblin_hut",
    "Lumberjack": "lumberjack",
    "Lava Hound": "hound", # usually mapped to hound
    "Inferno Dragon": "inferno_dragon",
    "Inferno Tower": "inferno_tower",
    "Bomb Tower": "bomb_tower",
    "Flying Machine": "flying_machine",
    "Battle Healer": "battle_healer",
    "Battle Ram": "battle_ram",
    "Ram Rider": "ram_rider",
    "Wall Breakers": "wall_breakers",
    "Bandit": "bandit",
    "Fisherman": "fisherman",
    "Mother Witch": "mother_witch",
    "Night Witch": "night_witch",
    "Dark Prince": "dark_prince",
    "Golden Knight": "golden_knight",
    "Archer Queen": "archer_queen",
    "Mighty Miner": "mighty_miner",
    "Monk": "monk",
    "Fire Spirit": "fire_spirit",
    "Heal Spirit": "heal_spirit",
    "Elixir Golem": "elixir_golem",
    "Goblin Drill": "goblin_drill",
    "Goblin Brawler": "goblin_brawler",
    "Sparky": "sparky",
    "The Log": "log",
}

def normalize_card_name(api_name: str) -> str:
    """Normalize the API card name to match pyclashbot card IDs."""
    if api_name in API_TO_CARD_ID_MAP:
        return API_TO_CARD_ID_MAP[api_name]
    # Default generic conversion: "Battle Ram" -> "battle_ram"
    return api_name.lower().replace(" ", "_").replace(".", "").replace("-", "")

def fetch_player_deck(player_tag: str, token: str) -> List[str]:
    """Fetch the player's active deck from the Clash Royale API."""
    if not player_tag or not token:
        return []
        
    encoded_tag = player_tag.replace("#", "%23")
    url = f"https://api.clashroyale.com/v1/players/{encoded_tag}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    # Ignore SSL certificate errors just in case
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            current_deck = data.get("currentDeck", [])
            card_ids = [normalize_card_name(card.get("name", "")) for card in current_deck]
            return card_ids
    except urllib.error.URLError as e:
        print(f"API Error fetching deck: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching deck: {e}")
        return []
