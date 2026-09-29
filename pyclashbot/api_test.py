import urllib.request
import json
import ssl

def get_player_info(player_tag: str, token: str) -> dict:
    # URL encode the player tag (replace # with %23)
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
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except urllib.error.URLError as e:
        print(f"API Error: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode('utf-8'))
        return {}

if __name__ == "__main__":
    TOKEN = "DUMMY_TOKEN"
    TAG = "#VUPVGYL0Q"
    info = get_player_info(TAG, TOKEN)
    
    print(f"Name: {info.get('name')}")
    print(f"Arena: {info.get('arena', {}).get('name')}")
    
    current_deck = info.get("currentDeck", [])
    print("\nCurrent Deck:")
    for card in current_deck:
        print(f"- {card.get('name')} (Lvl {card.get('level')})")
