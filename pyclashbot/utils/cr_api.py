"""Clash Royale Official API client.

Connects to the Supercell Clash Royale API (https://api.clashroyale.com/v1)
using the user's API token to fetch player profiles, current 8-card battle decks,
upcoming chest cycles, card statistics, and verified battle logs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pyclashbot.utils.caching import USER_SETTINGS_CACHE
from pyclashbot.utils.platform import get_app_data_dir

DEFAULT_KEY_FILE = Path(r"C:\Users\Gökçen\Desktop\clashroyaleapi.txt")
BASE_URL = "https://api.clashroyale.com/v1"

# Card name normalization to match pyclashbot card IDs
NAME_EXCEPTIONS = {
    "the log": "log",
    "mini p.e.k.k.a": "mini_pekka",
    "p.e.k.k.a": "pekka",
    "x-bow": "x_bow",
    "royal giant": "royal_giant",
    "battle ram": "battle_ram",
    "bomb tower": "bomb_tower",
    "goblin cage": "goblin_cage",
    "skeleton barrel": "skeleton_barrel",
    "wall breakers": "wall_breakers",
    "royal delivery": "royal_delivery",
    "flying machine": "flying_machine",
    "elixir collector": "elixir_collector",
    "elixir golem": "elixir_golem",
    "barbarian barrel": "barbarian_barrel",
    "skeleton army": "skeleton_army",
    "goblin barrel": "goblin_barrel",
    "goblin gang": "goblin_gang",
    "goblin drill": "goblin_drill",
    "goblin hut": "goblin_hut",
    "spear goblins": "spear_goblins",
    "dart goblin": "dart_goblin",
}


def normalize_card_name(name: str) -> str:
    """Normalize a Supercell API card name to a pyclashbot card ID."""
    clean = name.strip().lower()
    if clean in NAME_EXCEPTIONS:
        return NAME_EXCEPTIONS[clean]
    return clean.replace("-", "_").replace(".", "").replace(" ", "_")


def resolve_api_key(explicit_key: str | None = None) -> str:
    """Resolve API key from explicit arg, user cache, environment, or desktop file."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    cached = USER_SETTINGS_CACHE.get("cr_api_key")
    if cached and str(cached).strip():
        return str(cached).strip()

    env_key = os.environ.get("CR_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    if DEFAULT_KEY_FILE.is_file():
        try:
            content = DEFAULT_KEY_FILE.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError:
            pass

    return ""


def clean_tag(tag: str) -> str:
    """Format player or clan tag ensuring leading '#' and stripping spaces."""
    tag = tag.strip().upper().replace("O", "0")
    if not tag.startswith("#"):
        tag = "#" + tag
    return tag


class ClashRoyaleClient:
    """Client for querying the Supercell Clash Royale REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = resolve_api_key(api_key)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("No Clash Royale API key configured.")

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "PyClashBot/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(msg)
                reason = err_json.get("message", msg)
            except Exception:
                reason = msg
            raise RuntimeError(f"Clash Royale API error ({exc.code}): {reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to Clash Royale API: {exc}") from exc

    def get_cards(self) -> list[dict[str, Any]]:
        """Fetch all official Clash Royale card metadata."""
        data = self._get("cards")
        return data.get("items", [])  # type: ignore[no-any-return]

    def get_player(self, tag: str) -> dict[str, Any]:
        """Fetch full player profile for the given tag."""
        formatted_tag = urllib.parse.quote(clean_tag(tag))
        return self._get(f"players/{formatted_tag}")

    def get_player_deck(self, tag: str) -> list[str]:
        """Fetch the player's active 8-card deck normalized to pyclashbot card IDs."""
        profile = self.get_player(tag)
        current_deck = profile.get("currentDeck", [])
        return [normalize_card_name(card["name"]) for card in current_deck]

    def get_upcoming_chests(self, tag: str) -> list[dict[str, Any]]:
        """Fetch the player's upcoming chest rotation."""
        formatted_tag = urllib.parse.quote(clean_tag(tag))
        data = self._get(f"players/{formatted_tag}/upcomingchests")
        return data.get("items", [])  # type: ignore[no-any-return]

    def get_battle_log(self, tag: str) -> list[dict[str, Any]]:
        """Fetch the player's last 25 battles."""
        formatted_tag = urllib.parse.quote(clean_tag(tag))
        # battlelog returns a list directly
        if not self.api_key:
            raise ValueError("No Clash Royale API key configured.")
        url = f"{BASE_URL}/players/{formatted_tag}/battlelog"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "PyClashBot/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]

    def export_player_data(self, tag: str, dest_dir: Path | None = None) -> Path:
        """Export comprehensive player data (profile, deck, upcoming chests, battle log) to JSON."""
        tag = clean_tag(tag)
        if dest_dir is None:
            dest_dir = Path(get_app_data_dir("py-clash-bot")) / "exports"
        dest_dir.mkdir(parents=True, exist_ok=True)

        profile = self.get_player(tag)
        chests = self.get_upcoming_chests(tag)
        battles = self.get_battle_log(tag)

        data = {
            "player_tag": tag,
            "profile": profile,
            "active_deck_card_ids": [normalize_card_name(c["name"]) for c in profile.get("currentDeck", [])],
            "upcoming_chests": chests,
            "recent_battles": battles,
        }

        filename = f"player_{tag.replace('#', '')}_export.json"
        out_path = dest_dir / filename
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path
