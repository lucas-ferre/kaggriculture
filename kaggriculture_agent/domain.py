"""Small domain adapter. Rules audited against vendor/kaggriculture/COMMIT.txt.

No simulation seed, environment internals or opponent private state is read.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Crop:
    seed_cost: int
    first: int
    peak: int
    last: int
    peak_units: int
    ongoing: bool = False


# Game parameters from the Apache-2.0 Kaggle environment, see THIRD_PARTY_NOTICE.
CROPS = {
    "WHEAT": Crop(10, 2, 4, 4, 4),
    "CARROT": Crop(20, 2, 3, 3, 3),
    "TOMATO": Crop(50, 8, 11, 11, 4, True),
    "STRAWBERRY": Crop(100, 10, 16, 16, 4, True),
    "MELON": Crop(80, 10, 10, 12, 6),
}
PRODUCTS = tuple(CROPS) + ("EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = {
    "COW": {"cost": 400, "first": 8, "interval": 2, "product": "MILK", "cap": 6},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "product": "WOOL", "cap": 6},
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "product": "EGG", "cap": 4},
}
LAND_PRICES = (1000, 2000, 4000)
DEFAULT_GAME_CONFIG = {
    "boardSize": 10, "episodeSteps": 720, "turnsPerDay": 24,
    "shedCapacity": 100, "maxMarketOrdersPerTurn": 10,
    "farmHandCostMult": 1, "townShopUnlockInterval": 3,
    "townShopSellInterval": 4, "townCenterSellInterval": 24,
}


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shed_tiles(size):
    h = size // 2
    return ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


def nearest_shed(position, size):
    return min(shed_tiles(size), key=lambda p: (distance(position, p), p))


def move_towards(position, target):
    """Every in-bounds tile is passable, including occupied/locked tiles."""
    x, y = position
    tx, ty = target
    if x != tx:
        return ["EAST" if tx > x else "WEST"]
    if y != ty:
        return ["SOUTH" if ty > y else "NORTH"]
    return ["PASS"]


def fib_cost(hires, multiplier=1):
    a, b = 1, 1
    for _ in range(hires):
        a, b = b, a + b
    return a * multiplier


def game_configuration(configuration=None):
    result = dict(DEFAULT_GAME_CONFIG)
    if configuration:
        result.update(dict(configuration))
    # Never keep the environment's episode seed, even in local evaluations.
    result.pop("seed", None)
    return result
