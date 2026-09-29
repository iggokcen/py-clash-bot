"""Card facts the decision layer scores against.

Deliberately a **data table, not an ``if`` ladder**. Clash Royale ships ~200 cards
and adds more every season; hand-written per-card rules rot on every update. The
table is one row per card and the rules in :mod:`pyclashbot.bot.decide` are
generic over it.

Provenance of each field, and its known gaps:

* ``elixir`` / ``type`` from RoyaleAPI ``cr-api-data`` -- 100% coverage.
* ``hits_air`` / ``hits_ground`` from ``cards_stats_troop`` -- ~90% coverage.
* ``is_air`` is **hand-curated, because no API is reliable here.** All three
  candidate sources were checked against the real card list and each is wrong:
  ``cards[].is_air_unit`` flags only Battle Healer and Royal Ghost (both ground
  units), and ``cards_stats_troop[].height`` flags only Skeleton Barrel and
  Flying Machine, missing every actual flyer. The list below is the curated set.
* **Spells carry no RoyaleAPI stats at all** -- Fireball, Arrows, Rocket, The
  Log, Giant Snowball, Mirror, Goblin Barrel, Barbarian Barrel and Royal Delivery
  are absent from both ``key`` and ``name``. Those nine rows are hand-authored
  below, which is exactly the set the community reports the bot misusing.
* **Evolved and champion variants are entirely absent** -- RoyaleAPI ships zero
  ``is_evolved`` entries. Those rows are likewise hand-authored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Arena roles used by the scorer. One card has one primary role; that is enough
# to rank a hand and keeps the rule table small enough to reason about.
ROLE_BUILDING = "building"
ROLE_DEFENSE = "defense"
ROLE_SUPPORT = "support"
ROLE_WINCON = "wincon"
ROLE_SPELL = "spell"


@dataclass(frozen=True)
class CardFacts:
    card_id: str
    elixir: int
    role: str
    kind: str = "troop"  # troop | building | spell
    hits_air: bool = True
    hits_ground: bool = True
    is_air: bool = False
    is_spell: bool = False
    targets_buildings: bool = False
    splash: bool = False
    swarm: bool = False
    is_ranged: bool = False
    high_damage: bool = False
    notes: str = ""

    @property
    def tactical_summary(self) -> str:
        """Human-readable tactical profile for this card."""
        parts: list[str] = []
        if self.is_spell:
            parts.append("📦 Yer/Alan hedefli büyü")
        elif self.is_air:
            parts.append("🪽 Hava birimi")
        elif self.kind == "building":
            parts.append("🏛️ Savunma/Bina")
        else:
            parts.append("🟫 Yer birimi")

        if self.splash:
            parts.append("💥 Alan hasari")
        else:
            parts.append("🎯 Tek hedef")

        if not self.is_spell and self.kind != "building":
            if self.is_ranged:
                parts.append("🏹 Menzilli saldiri")
            else:
                parts.append("🤜 Yakin temas")

        if self.role == ROLE_WINCON:
            parts.append("⚔️ Ana hücum")
        elif self.role == ROLE_DEFENSE:
            parts.append("🛡️ Güçlü savunma")
        elif self.role == ROLE_SUPPORT:
            parts.append("⚔️ Savunma + Karsi saldiri")
        elif self.role == ROLE_SPELL:
            parts.append("⚡ Büyü destegi")

        if self.high_damage:
            parts.append("💥 Yüksek hasar")

        return " • ".join(parts)


def _f(card_id, elixir, role, **kw) -> CardFacts:
    kw.setdefault("is_spell", role == ROLE_SPELL or kw.get("kind") == "spell")
    return CardFacts(card_id=card_id, elixir=elixir, role=role, **kw)


# Cards with no RoyaleAPI row: the nine missing spells plus the evo/champion
# variants the API dropped. Values are the in-game costs; `splash`/`targets_buildings`
# are the behavioural facts the scorer needs, not frame data.
_HAND_AUTHORED: tuple[CardFacts, ...] = (
    _f("fireball", 4, ROLE_SPELL, kind="spell", splash=True, targets_buildings=True),
    _f("arrows", 3, ROLE_SPELL, kind="spell", splash=True, targets_buildings=True),
    _f("rocket", 6, ROLE_SPELL, kind="spell", splash=True, targets_buildings=True),
    _f("log", 2, ROLE_SPELL, kind="spell", hits_air=False, targets_buildings=True),
    _f("snowball", 3, ROLE_SPELL, kind="spell", targets_buildings=True),
    _f("goblin_barrel", 3, ROLE_SPELL, kind="spell", hits_air=False),
    _f("barb_barrel", 3, ROLE_SPELL, kind="spell", hits_air=False, targets_buildings=True),
    _f("royal_delivery", 3, ROLE_SPELL, kind="spell", hits_air=False),
    _f("clone", 3, ROLE_SPELL, kind="spell"),
    _f("zap", 2, ROLE_SPELL, kind="spell", hits_air=True, targets_buildings=True),
    _f("rage", 2, ROLE_SPELL, kind="spell"),
    _f("gob_curse", 2, ROLE_SPELL, kind="spell"),
    _f("void", 3, ROLE_SPELL, kind="spell"),
    _f("mirror", 0, ROLE_SPELL, kind="spell"),
    _f("freeze", 4, ROLE_SPELL, kind="spell"),
    _f("poison", 4, ROLE_SPELL, kind="spell", targets_buildings=True),
    _f("earthquake", 3, ROLE_SPELL, kind="spell", hits_air=False, targets_buildings=True),
    _f("lightning", 6, ROLE_SPELL, kind="spell", splash=True, targets_buildings=True),
    _f("tornado", 3, ROLE_SPELL, kind="spell"),
    # champion / evolution variants RoyaleAPI never shipped
    _f("hero_knight", 3, ROLE_DEFENSE, hits_air=False, is_air=False),
    _f("hero_barb_barrel", 3, ROLE_SPELL, kind="spell", hits_air=False),
    _f("hero_bowler", 4, ROLE_DEFENSE, hits_air=False),
    _f("hero_goblins", 3, ROLE_SUPPORT, hits_air=False, swarm=True),
    _f("hero_magic_archer", 5, ROLE_SUPPORT),
    _f("hero_mini_pekka", 4, ROLE_DEFENSE, hits_air=False),
    _f("hero_musketeer", 4, ROLE_SUPPORT),
    _f("hero_giant", 6, ROLE_WINCON, hits_air=False, targets_buildings=True),
    _f("evo_musketeer", 4, ROLE_SUPPORT),
    _f("evo_electro_dragon", 4, ROLE_DEFENSE, is_air=True),
    _f("evo_inferno_dragon", 4, ROLE_DEFENSE, is_air=True),
    _f("evo_minion_horde", 5, ROLE_DEFENSE, is_air=True, swarm=True),
    _f("evo_archers", 3, ROLE_SUPPORT),
    _f("evo_skeletons", 1, ROLE_DEFENSE, hits_air=False, swarm=True),
    _f("evo_cannon", 3, ROLE_BUILDING, kind="building", hits_air=False),
    _f("evo_skeleton_barrel", 3, ROLE_BUILDING, kind="building"),
    _f("evo_wizard", 5, ROLE_SPELL, kind="spell", splash=True),
    _f("evo_royal_hogs", 4, ROLE_WINCON, hits_air=False, targets_buildings=True),
    _f("evo_royal_ghost", 4, ROLE_DEFENSE, hits_air=False),
    _f("evo_valkyrie", 4, ROLE_DEFENSE, hits_air=False, splash=True),
    _f("evo_skeleton_army", 3, ROLE_DEFENSE, hits_air=False, swarm=True),
    _f("evo_wall_breakers", 2, ROLE_DEFENSE, hits_air=False, splash=True),
    _f("evo_pekka", 7, ROLE_DEFENSE, hits_air=False),
    _f("evo_royal_recruits", 5, ROLE_DEFENSE, hits_air=False, splash=True),
    _f("evo_witch", 5, ROLE_SPELL, kind="spell", splash=True),
    _f("evo_princess", 3, ROLE_SUPPORT),
    _f("evo_lumberjack", 4, ROLE_DEFENSE, hits_air=False, splash=True),
    _f("evo_executioner", 5, ROLE_DEFENSE, splash=True),
    _f("evo_goblin_drill", 4, ROLE_WINCON, hits_air=False, targets_buildings=True),
    _f("evo_goblin_giant", 6, ROLE_WINCON, hits_air=False, targets_buildings=True),
    _f("hero_balloon", 5, ROLE_WINCON, is_air=True, targets_buildings=True),
    _f("hero_dark_prince", 4, ROLE_DEFENSE, hits_air=False, splash=True),
    _f("hero_ice_golem", 2, ROLE_DEFENSE, hits_air=False),
    _f("hero_mega_minion", 3, ROLE_DEFENSE, is_air=True),
    _f("hero_tombstone", 3, ROLE_BUILDING, kind="building"),
    _f("hero_wizard", 6, ROLE_SPELL, kind="spell", splash=True),
)

# RoyaleAPI rows, joined from the derived card table (Step 0 output,
# `pcb-card-table/v0`). Populated lazily from the JSON shipped alongside the
# package so the table stays one source of truth.
_ROYALEAPI_FALLBACK: dict[str, tuple[int, str, bool, bool, bool]] = {
    # card_id: (elixir, kind, hits_air, hits_ground, is_air)
    "hog": (4, "troop", False, True, False),
    "royal_hogs": (5, "troop", False, True, False),
    "battle_ram": (4, "troop", False, True, False),
    "ram_rider": (5, "troop", False, True, False),
    "wall_breakers": (3, "troop", False, True, False),
    "barbarian_hut": (5, "building", False, True, False),
    "tombstone": (3, "building", False, True, False),
    "cannon": (3, "building", False, True, False),
    "tesla": (4, "building", True, True, False),
    "inferno_tower": (5, "building", True, True, False),
    "bomb_tower": (4, "building", False, True, False),
    "mortar": (3, "building", False, True, False),
    "xbow": (6, "building", False, True, False),
    "hut": (5, "building", False, True, False),
    "goblin_hut": (5, "building", False, True, False),
    "giant": (5, "troop", False, True, False),
    "goblin_giant": (6, "troop", False, True, False),
    "wizard": (5, "troop", True, True, False),
    "witch": (5, "troop", True, True, False),
    "night_witch": (5, "troop", True, True, False),
    "mother_witch": (7, "troop", True, True, False),
    "three_musketeers": (4, "troop", True, True, False),
    "archers": (3, "troop", True, True, False),
    "firecracker": (4, "troop", True, True, False),
    "baby_dragon": (4, "troop", True, True, True),
    "inferno_dragon": (4, "troop", True, True, True),
    "flying_machine": (4, "troop", True, True, True),
    "phoenix": (4, "troop", True, True, True),
    "electro_dragon": (4, "troop", True, True, True),
    "electro_spirit": (1, "troop", True, True, False),
    "electro_wizard": (5, "troop", True, True, False),
    "dark_prince": (4, "troop", False, True, False),
    "mini_pekka": (4, "troop", False, True, False),
    "pekka": (7, "troop", False, True, False),
    "bandit": (3, "troop", False, True, False),
    "fisherman": (4, "troop", False, True, False),
    "princess": (4, "troop", True, True, False),
    "archer_queen": (5, "troop", True, True, False),
    "ice_wizard": (3, "troop", True, True, False),
    "valkyrie": (4, "troop", False, True, False),
    "knight": (3, "troop", False, True, False),
    "lumberjack": (4, "troop", False, True, False),
    "executioner": (5, "troop", True, True, False),
    "royal_ghost": (3, "troop", False, True, False),
    "void": (3, "troop", True, True, True),
    "little_prince": (3, "troop", True, True, False),
    "miner": (3, "troop", False, True, False),
    "goblin_drill": (4, "troop", False, True, False),
    "goblin_machine": (6, "troop", False, True, False),
    "battle_healer": (4, "troop", True, True, False),
    "monk": (3, "troop", True, True, False),
    "ice_spirit": (1, "troop", True, True, False),
    "fire_spirit": (1, "troop", True, True, False),
    "heal_spirit": (2, "troop", True, True, False),
    "zappies": (4, "troop", True, True, False),
    "bats": (2, "troop", True, True, True),
    "minion_horde": (5, "troop", True, True, True),
    "minions": (3, "troop", True, True, True),
    "mega_minion": (3, "troop", True, True, True),
    "dart_goblin": (3, "troop", True, True, False),
    "spear_goblins": (2, "troop", True, True, False),
    "goblins": (2, "troop", False, True, False),
    "goblin_gang": (3, "troop", False, True, False),
    "skeletons": (1, "troop", False, True, False),
    "skeleton_army": (3, "troop", False, True, False),
    "skeleton_barrel": (3, "building", False, True, False),
    "skeleton_dragons": (4, "troop", True, True, True),
    "graveyard": (3, "spell", False, True, False),
    "poison": (4, "spell", False, True, False),
    "arrows": (3, "spell", True, True, False),
    "mighty_miner": (4, "troop", False, True, False),
    "royal_delivery": (3, "spell", False, True, False),
    "goblin_cage": (4, "building", False, True, False),
    "furnace": (5, "building", True, True, False),
    "elixir_collector": (5, "building", False, True, False),
    "suspicious_bush": (3, "building", False, True, False),
    "shield_wall": (4, "building", False, True, False),
    "bomb": (3, "building", False, True, False),
    "elixir_golem": (3, "troop", False, True, False),
    "golem": (8, "troop", False, True, False),
    "giant_skeleton": (6, "troop", False, True, False),
    "royal_giant": (6, "troop", False, True, False),
    "electro_giant": (5, "troop", False, True, False),
    "ice_golem": (2, "troop", False, True, False),
    "mega_knight": (7, "troop", False, True, False),
    "prince": (5, "troop", False, True, False),
    "golden_knight": (4, "troop", False, True, False),
    "berserker": (5, "troop", False, True, False),
    "boss_bandit": (5, "troop", False, True, False),
    "goblinstein": (6, "troop", False, True, False),
    "skeleton_king": (4, "troop", False, True, False),
    "royal_recruits": (5, "troop", False, True, False),
    "barbarians": (5, "troop", False, True, False),
    "minion_giant": (6, "troop", False, True, False),
    "cannon_cart": (4, "troop", False, True, False),
    "bowler": (4, "troop", False, True, False),
    "hunter": (4, "troop", True, True, False),
    "lava_hound": (7, "troop", False, True, True),
    "balloon": (5, "troop", False, True, True),
    "hog_rider": (4, "troop", False, True, False),
    "bomber": (2, "troop", False, True, False),
    "elite_barbarians": (6, "troop", False, True, False),
    "guards": (3, "troop", False, True, False),
    "magic_archer": (4, "troop", True, True, False),
    "musketeer": (4, "troop", True, True, False),
    "rascals": (5, "troop", True, True, False),
    "sparky": (6, "troop", False, True, False),
    "goblin_demolisher": (4, "troop", False, True, False),
    "rune_giant": (5, "troop", False, True, False),
    "spirit_empress": (6, "troop", True, True, False),
}

# The curated air roster. No API provides this correctly (see the module
# docstring), so it is written out explicitly and asserted by a test, so that
# adding a card to the table cannot quietly change which units the air-defence
# rule will accept.
#
# Sources reconciled: CR release notes plus KataCR's ``flying_unit_list``
# (wty-yy/KataCR, MIT), which is a *detector class* list rather than a gameplay
# one -- it also contains spells (fireball, zap, arrows) and projectiles (axe,
# goblin_ball, phoenix), so it is a cross-check, not an authority.
AIR_UNITS = frozenset(
    {
        "balloon",
        "bats",
        "bat_evolution",
        "baby_dragon",
        "electro_dragon",
        "flying_machine",
        "inferno_dragon",
        "lava_hound",
        "mega_minion",
        "minion_horde",
        "minions",
        "phoenix",
        "skeleton_dragons",
        "evo_baby_dragon",
        "evo_bats",
        "evo_electro_dragon",
        "evo_inferno_dragon",
        "evo_minion_horde",
        "evo_skeleton_dragons",
        "hero_balloon",
        "hero_mega_minion",
    }
)


_ROLE_BY_KIND = {
    "spell": ROLE_SPELL,
    "building": ROLE_BUILDING,
}

# Win conditions and the other roles the API does not carry. Without an explicit
# list a Hog Rider scores as plain defence and walks to the back line instead of
# the bridge, which is a real placement bug rather than a scoring nicety.
_WIN_CONDITIONS = frozenset(
    {
        "hog",
        "royal_hogs",
        "battle_ram",
        "ram_rider",
        "giant",
        "goblin_giant",
        "royal_giant",
        "electro_giant",
        "giant_skeleton",
        "golem",
        "balloon",
        "hero_balloon",
        "hero_giant",
        "goblin_drill",
        "evo_goblin_drill",
        "evo_goblin_giant",
        "evo_royal_hogs",
        "miner",
        "goblin_machine",
        "pekka",
        "prince",
        "night_witch",
        "elite_barbarians",
        "sparky",
        "rune_giant",
    }
)

_ROLE_OVERRIDES: dict[str, str] = {cid: ROLE_WINCON for cid in _WIN_CONDITIONS}
_ROLE_OVERRIDES.update(
    {
        "wizard": ROLE_SPELL,
        "mother_witch": ROLE_SPELL,
        "three_musketeers": ROLE_SUPPORT,
        "archers": ROLE_SUPPORT,
        "firecracker": ROLE_SUPPORT,
        "baby_dragon": ROLE_SUPPORT,
        "inferno_dragon": ROLE_SUPPORT,
        "electro_dragon": ROLE_SUPPORT,
        "electro_spirit": ROLE_DEFENSE,
        "void": ROLE_DEFENSE,
        "princess": ROLE_SUPPORT,
        "archer_queen": ROLE_SUPPORT,
        "heal_spirit": ROLE_SUPPORT,
        "battle_healer": ROLE_SUPPORT,
        "graveyard": ROLE_SPELL,
        "elixir_collector": ROLE_BUILDING,
        "bomber": ROLE_DEFENSE,
        "guards": ROLE_DEFENSE,
        "magic_archer": ROLE_SUPPORT,
        "musketeer": ROLE_SUPPORT,
        "rascals": ROLE_DEFENSE,
        "goblin_demolisher": ROLE_DEFENSE,
        "spirit_empress": ROLE_SUPPORT,
    }
)

_TABLE: dict[str, CardFacts] = {}
for _c in _HAND_AUTHORED:
    _TABLE[_c.card_id] = _c
for _cid, (_el, _kind, _ha, _hg, _air) in _ROYALEAPI_FALLBACK.items():
    if _cid in _TABLE:
        continue
    _TABLE[_cid] = CardFacts(
        card_id=_cid,
        elixir=_el,
        role=_ROLE_OVERRIDES.get(_cid) or _ROLE_BY_KIND.get(_kind, ROLE_DEFENSE),
        kind=_kind,
        hits_air=_ha,
        hits_ground=_hg,
        is_air=_air or _cid in AIR_UNITS,
        is_spell=_kind == "spell",
        targets_buildings=_kind == "spell",
        swarm=_cid in {"skeletons", "goblins", "minions", "bats", "zappies", "goblin_gang", "guards"},
    )

RANGED_UNITS = frozenset(
    {
        "archers",
        "spear_goblins",
        "dart_goblin",
        "princess",
        "magic_archer",
        "musketeer",
        "three_musketeers",
        "wizard",
        "ice_wizard",
        "electro_wizard",
        "firecracker",
        "hunter",
        "bowler",
        "executioner",
        "witch",
        "mother_witch",
        "minions",
        "minion_horde",
        "mega_minion",
        "flying_machine",
        "baby_dragon",
        "inferno_dragon",
        "electro_dragon",
        "skeleton_dragons",
        "phoenix",
        "archer_queen",
        "little_prince",
        "sparky",
        "bomber",
        "fire_spirit",
        "ice_spirit",
        "heal_spirit",
        "electro_spirit",
        "evo_archers",
        "evo_wizard",
        "evo_firecracker",
        "evo_bomber",
        "evo_musketeer",
        "evo_minion_horde",
        "hero_musketeer",
        "hero_magic_archer",
    }
)

HIGH_DAMAGE_UNITS = frozenset(
    {
        "mini_pekka",
        "pekka",
        "prince",
        "sparky",
        "hunter",
        "inferno_dragon",
        "mighty_miner",
        "elite_barbarians",
        "archer_queen",
        "hero_mini_pekka",
    }
)

SPLASH_UNITS = frozenset(
    {
        "valkyrie",
        "wizard",
        "baby_dragon",
        "bomber",
        "dark_prince",
        "mega_knight",
        "firecracker",
        "executioner",
        "bowler",
        "ice_wizard",
        "electro_dragon",
        "skeleton_dragons",
        "princess",
        "magic_archer",
        "sparky",
        "royal_delivery",
        "fireball",
        "arrows",
        "rocket",
        "snowball",
        "zap",
        "poison",
        "earthquake",
        "lightning",
        "tornado",
        "log",
        "barb_barrel",
        "evo_valkyrie",
        "evo_wizard",
        "evo_bomber",
        "evo_firecracker",
    }
)

# The curated roster is the source of truth, applied over the row above so a row
# that forgot is_air cannot downgrade an air unit to ground.
for _cid in AIR_UNITS & _TABLE.keys():
    _row = _TABLE[_cid]
    _TABLE[_cid] = replace(_row, is_air=True, hits_air=True)

# Apply tactical properties (ranged, high damage, splash) across all registered cards
for _cid, _row in list(_TABLE.items()):
    _TABLE[_cid] = replace(
        _row,
        splash=_cid in SPLASH_UNITS or _row.splash or any(_cid.endswith(s) for s in SPLASH_UNITS),
        is_ranged=_cid in RANGED_UNITS or any(_cid.endswith(r) for r in RANGED_UNITS),
        high_damage=_cid in HIGH_DAMAGE_UNITS or any(_cid.endswith(h) for h in HIGH_DAMAGE_UNITS),
    )


_ALIASES = {
    "the_log": "log",
    "hog_rider": "hog",
    "barbarian_barrel": "barb_barrel",
    "giant_snowball": "snowball",
    "barb_hut": "barbarian_hut",
    "fire_cracker": "firecracker",
    "evo_fire_cracker": "evo_firecracker",
}


def facts_for(card_id: str) -> CardFacts:
    """Look up a card, falling back to a conservative defensive troop.

    An unknown id must never crash a fight, and must never be assumed to be a
    spell: a wrong guess here would send a troop to the tower line.
    """
    card_id = _ALIASES.get(card_id, card_id)
    known = _TABLE.get(card_id)
    if known is not None:
        return known
    base_id = card_id.removeprefix("evo_").removeprefix("hero_")
    if base_id in _TABLE:
        return replace(_TABLE[base_id], card_id=card_id)
    return CardFacts(card_id=card_id, elixir=4, role=ROLE_DEFENSE, notes="unknown card")


def known_card_ids() -> tuple[str, ...]:
    return tuple(sorted(_TABLE))


TABLE_SIZE = len(_TABLE)
