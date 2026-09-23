"""Public behavioral estimates and an optional episode-local contextual bandit.

Opponent orders/seeds/shed are not observable. Profiles and stock are uncertain
estimates, never identification of a particular downloaded bot or known tape.
"""
from collections import Counter
import hashlib
import json
import math
import random

from .domain import ANIMALS, CROPS, PRODUCTS


class OpponentModel:
    def __init__(self):
        self.previous = None
        self.stock = {p: 0. for p in PRODUCTS}
        self.profile = {}

    def update(self, obs, market):
        step = int(obs.get("step", 0))
        player = int(obs["player"])
        if self.previous and (step <= self.previous["step"] or player != self.previous["player"]):
            self.__init__()
        farm = obs["farms"][1 - player]
        tiles = {(x, y): {k: tile[k] for k in ("kind", "crop", "animal", "planted_day", "placed_day", "yield_units") if k in tile}
                 for y, row in enumerate(farm.get("tiles", [])) for x, tile in enumerate(row)
                 if isinstance(tile, dict)}
        crops = Counter(t.get("crop") for t in tiles.values() if t.get("kind") == "PLANT")
        animals = Counter(t.get("animal") for t in tiles.values() if t.get("animal") in ANIMALS)
        ready = Counter()
        day = int(obs["day"])
        for tile in tiles.values():
            animal, crop = tile.get("animal"), tile.get("crop")
            product = ANIMALS[animal]["product"] if animal in ANIMALS else crop
            if product in PRODUCTS and (animal in ANIMALS or day - tile.get("planted_day", day) >= CROPS[crop].first):
                ready[product] += max(0, tile.get("yield_units", 0))
        if self.previous and step == self.previous["step"] + 1:
            # Intraday disappearing yield is a harvest signal, with no claim
            # that transport/sale occurred or that unseen starting stock is zero.
            if day == self.previous["day"]:
                for pos, old in self.previous["tiles"].items():
                    new = tiles.get(pos)
                    animal, crop = old.get("animal"), old.get("crop")
                    product = ANIMALS[animal]["product"] if animal in ANIMALS else crop
                    if product not in PRODUCTS or new and new.get("kind") == "WEED":
                        continue
                    same = (new and old.get("animal") == new.get("animal") and
                            old.get("crop") == new.get("crop") and old.get("planted_day") == new.get("planted_day"))
                    if same or new is None:
                        loss = max(0, old.get("yield_units", 0) - (new or {}).get("yield_units", 0))
                        self.stock[product] += .8 * loss
            for product in PRODUCTS:
                history = market._history.get(product, ())
                if history and not history[-1][1]:
                    self.stock[product] = max(0., self.stock[product] - max(0., history[-1][0]))
                self.stock[product] *= .995
        activity = sum(crops.values()) + sum(animals.values())
        concentration = max(crops.values(), default=0) / max(1, sum(crops.values()))
        scores = {"passive": 1 / (1 + activity),
                  "livestock": 1 + 2 * sum(animals.values()),
                  "crop_flood": 1 + sum(crops.values()) * concentration,
                  "diversified": 1 + 3 * len(crops) + min(sum(crops.values()), sum(animals.values()))}
        total = sum(scores.values())
        probabilities = {key: value / total for key, value in scores.items()}
        self.profile = {"probabilities": probabilities, "dominant": max(probabilities, key=probabilities.get),
                        "visible_crops": dict(crops), "visible_animals": dict(animals),
                        "visible_ready": dict(ready), "inferred_stock": dict(self.stock),
                        "public_money": float(farm.get("money", 0)),
                        "confidence": "behavioral_heuristic_not_bot_identity"}
        self.previous = {"step": step, "player": player, "day": day, "tiles": tiles}
        return self.profile

    def supply_risk(self, product):
        return self.stock.get(product, 0.) + self.profile.get("visible_ready", {}).get(product, 0)


class ContextualBandit:
    """Gaussian linear Thompson sampling; opt-in pending arena evidence.

    Daily margin changes update the previously chosen arm. Rewards are delayed
    and confounded by past investments, so this is an experimental controller,
    not calibrated policy success probability. No cross-episode data is saved.
    """
    ARMS = ("adaptive", "melon", "cashflow", "contrarian")
    DIMENSION = 6

    def __init__(self):
        n = self.DIMENSION
        self.inverse = {a: [[float(i == j) / 4 for j in range(n)] for i in range(n)] for a in self.ARMS}
        self.b = {a: [0.] * n for a in self.ARMS}
        self.counts = Counter()
        self.arm, self.context = "adaptive", None
        self.last_day = self.selected_day = None
        self.last_margin = 0.

    def _update(self, arm, x, reward):
        inv = self.inverse[arm]
        ax = [sum(v * z for v, z in zip(row, x)) for row in inv]
        denominator = 1 + sum(a * z for a, z in zip(ax, x))
        self.inverse[arm] = [[inv[i][j] - ax[i] * ax[j] / denominator for j in range(len(x))]
                             for i in range(len(x))]
        self.b[arm] = [v + reward * z for v, z in zip(self.b[arm], x)]
        self.counts[arm] += 1

    def choose(self, obs, profile):
        day, player = int(obs["day"]), int(obs["player"])
        mine, rival = obs["farms"][player], obs["farms"][1 - player]
        margin = float(mine["money"] - rival["money"])
        if self.last_day is not None and day < self.last_day:
            self.__init__()
        x = [1., min(1., day / 29), min(1., mine["money"] / 10000),
             min(1., sum(profile.get("visible_animals", {}).values()) / 12),
             min(1., max(profile.get("visible_crops", {}).values(), default=0) / 25),
             min(1., len(obs.get("town", {}).get("unlocked_shops", [])) / 8)]
        if self.last_day != day:
            if self.context is not None:
                self._update(self.arm, self.context, math.tanh((margin - self.last_margin) / 5000))
            self.last_margin, self.last_day = margin, day
        if self.selected_day is not None and day - self.selected_day < 3:
            return self.arm
        seed_state = json.dumps([day, player, x, dict(self.counts)], sort_keys=True).encode()
        rng = random.Random(int.from_bytes(hashlib.blake2b(seed_state, digest_size=8).digest(), "big"))
        samples = {}
        for arm in self.ARMS:
            inv, n = self.inverse[arm], len(x)
            mean = [sum(v * z for v, z in zip(row, self.b[arm])) for row in inv]
            chol = [[0.] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    value = inv[i][j] - sum(chol[i][k] * chol[j][k] for k in range(j))
                    chol[i][j] = math.sqrt(max(1e-12, value)) if i == j else value / chol[j][j]
            noise = [rng.gauss(0, .2) for _ in range(n)]
            theta = [mean[i] + sum(chol[i][j] * noise[j] for j in range(i + 1)) for i in range(n)]
            samples[arm] = sum(a * z for a, z in zip(theta, x))
        self.arm = max(self.ARMS, key=lambda a: samples[a])
        self.context, self.selected_day = x, day
        return self.arm
