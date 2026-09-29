import json
from unittest.mock import MagicMock, patch

import pytest

from pyclashbot.utils.cr_api import (
    ClashRoyaleClient,
    clean_tag,
    normalize_card_name,
    resolve_api_key,
)


def test_clean_tag():
    assert clean_tag("8P92UQLR") == "#8P92UQLR"
    assert clean_tag("#8p92uqlr") == "#8P92UQLR"
    assert clean_tag("  #abc  ") == "#ABC"


def test_normalize_card_name():
    assert normalize_card_name("Knight") == "knight"
    assert normalize_card_name("The Log") == "log"
    assert normalize_card_name("Mini P.E.K.K.A") == "mini_pekka"
    assert normalize_card_name("X-Bow") == "x_bow"
    assert normalize_card_name("Royal Giant") == "royal_giant"
    assert normalize_card_name("Barbarian Barrel") == "barbarian_barrel"


def test_resolve_api_key_explicit():
    assert resolve_api_key("my-token-123") == "my-token-123"


def test_clash_royale_client_unconfigured():
    client = ClashRoyaleClient(api_key="")
    # Clear env/file key if any
    client.api_key = ""
    with pytest.raises(ValueError, match="No Clash Royale API key"):
        client.get_cards()


def test_clash_royale_client_mock_player():
    client = ClashRoyaleClient(api_key="mock_key")
    mock_payload = {
        "tag": "#8P92UQLR",
        "name": "KingArthur",
        "trophies": 7500,
        "currentDeck": [
            {"name": "Knight"},
            {"name": "Archers"},
            {"name": "Fireball"},
            {"name": "The Log"},
            {"name": "Hog Rider"},
            {"name": "Cannon"},
            {"name": "Ice Spirit"},
            {"name": "Skeletons"},
        ],
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        profile = client.get_player("#8P92UQLR")
        assert profile["name"] == "KingArthur"
        assert profile["trophies"] == 7500

        deck = client.get_player_deck("#8P92UQLR")
        assert deck == [
            "knight",
            "archers",
            "fireball",
            "log",
            "hog_rider",
            "cannon",
            "ice_spirit",
            "skeletons",
        ]
