"""Version 0.1.1 physical-action regressions against the pinned engine."""
import copy
from types import SimpleNamespace
import unittest

from kaggriculture_agent.domain import game_configuration, distance, nearest_shed
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from kaggriculture_agent.scheduler import TaskScheduler
from scripts.benchmark import load_engine, make_environment

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine()

    def setUp(self):
        self.env = make_environment(9, 720, {"weedSpawnChance": 0})
        self.config = game_configuration()
        self.scheduler = TaskScheduler(PolicyConfig())

    @property
    def obs(self):
        return self.env.state[0].observation

    def execute(self, plan, market=None):
        before = copy.deepcopy(self.obs)
        actions, predicted, debug = self.scheduler.actions(self.obs, self.config, plan)
        self.assertEqual(self.obs, before)
        self.env.step([{"farmer": actions[0], "hands": actions[1:],
                        "market": market or []}, PASS])
        return actions, predicted, debug

    def test_hour_22_plant_reserves_same_worker_water_even_with_new_urgency(self):
        while self.obs.step < 22:
            self.env.step([PASS, PASS])
        self.obs.private["seeds"]["WHEAT"] = 1
        plan = CropPlan("WHEAT", None, 1, 0)
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["PLANT", "WHEAT"])
        self.assertIn(0, self.scheduler.water_reservations)
        # A newly available premium harvest may not steal the reserved worker.
        self.obs.farms[0]["tiles"][3][4] = self.engine._new_animal("COW", 0)
        self.obs.farms[0]["tiles"][3][4].update(yield_units=6, fed_today=True)
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["WATER"])
        self.assertEqual(self.obs.hour, 0)
        self.assertEqual(self.obs.farms[0]["tiles"][4][4]["kind"], "PLANT")
        self.assertEqual(self.obs.farms[0]["tiles"][4][4]["consecutive_unwatered"], 0)

    def test_failed_plant_reservation_does_not_force_invalid_water(self):
        self.obs.private["seeds"]["WHEAT"] = 1
        plan = CropPlan("WHEAT", None, 1, 0)
        self.execute(plan)
        self.obs.farms[0]["tiles"][4][4] = None
        actions, _, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["PASS"])
        self.assertEqual(self.scheduler.water_reservations, {})

    def test_cold_start_recovers_water_from_observation(self):
        self.obs.private["seeds"]["WHEAT"] = 1
        plan = CropPlan("WHEAT", None, 1, 0)
        self.execute(plan)
        self.scheduler = TaskScheduler(PolicyConfig())
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["WATER"])
        self.assertTrue(self.obs.farms[0]["tiles"][4][4]["watered_today"])

    def test_zero_animal_targets_leave_crop_core_available(self):
        self.obs.private["seeds"]["WHEAT"] = 1
        plan = SimpleNamespace(targets={"WHEAT": 1},
                               animal_targets={"COW": 0, "SHEEP": 0, "GOOSE": 0})
        actions, _, debug = self.execute(plan)
        self.assertEqual(actions[0], ["PLANT", "WHEAT"])
        self.assertEqual(debug["animal_sites"], 0)

    def test_reset_removes_worker_commitments_and_daily_rehire_routes(self):
        self.scheduler.water_reservations[6] = ((4, 4), "WHEAT", 0, 1)
        self.scheduler.targets[6] = ("FEED", (4, 4), "COW", -1)
        self.scheduler.reset()
        self.assertFalse(self.scheduler.water_reservations)
        self.assertFalse(self.scheduler.targets)

    def test_three_crop_plan_is_executed_without_legacy_pair(self):
        self.obs.private["seeds"].update(WHEAT=1, CARROT=1, MELON=1)
        plan = SimpleNamespace(targets={"WHEAT": 1, "CARROT": 1, "MELON": 1},
                               animal_targets={})
        for _ in range(16):
            self.execute(plan)
        crops = {t["crop"] for row in self.obs.farms[0]["tiles"] for t in row
                 if isinstance(t, dict) and t.get("kind") == "PLANT"}
        self.assertEqual(crops, set(plan.targets))

    def test_feed_uses_pickup_inventory_then_care_and_fertilizer(self):
        tile = self.engine._new_animal("COW", 0)
        tile["fertilizer_available"] = True
        self.obs.farms[0]["tiles"][4][4] = tile
        self.obs.private["shed"]["WHEAT"] = 1
        plan = SimpleNamespace(targets={}, animal_targets={"COW": 1})
        actions, predicted, _ = self.execute(plan)
        self.assertEqual(actions[0], ["PICKUP", "WHEAT", 1])
        self.assertEqual(predicted["WHEAT"], 0)
        self.assertEqual(self.obs.private["inventories"][0]["WHEAT"], 1)
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["FEED"])
        self.assertTrue(self.obs.farms[0]["tiles"][4][4]["fed_today"])
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["CARE"])
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["COLLECT_FERTILIZER"])
        self.assertEqual(self.obs.private["inventories"][0]["FERTILIZER"], 1)

    def test_shed_ledger_reserves_parallel_feed_pickups(self):
        farm, private = self.obs.farms[0], self.obs.private
        farm["hands"] = [[4, 4], [4, 4]]
        private["inventories"] = [{}, {}, {}]
        private["shed"]["WHEAT"] = 2
        for x, y in [(4, 4), (4, 3), (3, 4)]:
            farm["tiles"][y][x] = self.engine._new_animal("COW", 0)
        actions, predicted, _ = self.execute(SimpleNamespace(targets={}, animal_targets={}))
        pickups = [a[2] for a in actions if a[0] == "PICKUP" and a[1] == "WHEAT"]
        self.assertEqual(sum(pickups), 2)
        self.assertEqual(predicted["WHEAT"], 0)
        self.assertEqual(sum(i.get("WHEAT", 0) for i in self.obs.private["inventories"]), 2)

    def test_remote_feed_inventory_does_not_stall_nearby_worker(self):
        farm, private = self.obs.farms[0], self.obs.private
        farm["hands"] = [[0, 0]]
        private["inventories"] = [{}, {"WHEAT": 1}]
        private["shed"]["WHEAT"] = 1
        farm["tiles"][4][4] = self.engine._new_animal("COW", 0)
        actions, predicted, _ = self.execute(SimpleNamespace(targets={}, animal_targets={}))
        self.assertEqual(actions[0], ["PICKUP", "WHEAT", 1])
        self.assertEqual(predicted["WHEAT"], 0)

    def test_installation_repairs_weed_and_places_shed_animal(self):
        self.obs.farms[0]["tiles"][4][4] = {"kind": "WEED"}
        self.obs.private["shed"].update(COW=1, WHEAT=2)
        plan = SimpleNamespace(targets={}, animal_targets={"COW": 1})
        seen = []
        for _ in range(8):
            actions, _, _ = self.execute(plan)
            seen.append(actions[0])
        for expected in [["PICKUP", "COW", 1], ["DIG"], ["BUILD_PASTURE"], ["PLACE", "COW"], ["FEED"], ["CARE"]]:
            self.assertIn(expected, seen)
        tile = self.obs.farms[0]["tiles"][4][4]
        self.assertEqual(tile["animal"], "COW")
        self.assertTrue(tile["fed_today"])
        self.assertTrue(tile["cared_today"])

    def test_twelve_animals_receive_sustainable_daily_feed_and_care(self):
        farm, private = self.obs.farms[0], self.obs.private
        # Compact core spans the first bought quadrant as in actual expansion.
        positions = sorted(((x, y) for y in range(5) for x in range(10)),
                           key=lambda p: (distance(p, nearest_shed(p, 10)), p[1], p[0]))[:12]
        for i, (x, y) in enumerate(positions):
            farm["tiles"][y][x] = self.engine._new_animal("COW" if i < 8 else "SHEEP", 0)
        plan = SimpleNamespace(targets={}, animal_targets={"COW": 8, "SHEEP": 4})
        for _ in range(4 * 24):
            if self.obs.hour == 0:
                self.obs.private["shed"]["WHEAT"] = 12
                # Recreate the workforce that market hiring supplies each day.
                self.obs.farms[0]["hands"] = [[4, 4], [5, 4], [4, 5], [5, 5], [4, 4], [5, 4], [4, 5]]
                self.obs.private["inventories"] = [{} for _ in range(8)]
            if self.obs.hour == 23:
                tiles = [t for row in self.obs.farms[0]["tiles"] for t in row
                         if isinstance(t, dict) and t.get("animal")]
                self.assertEqual(len(tiles), 12)
                self.assertTrue(all(t["fed_today"] for t in tiles))
                self.assertTrue(all(t["cared_today"] for t in tiles))
            self.execute(plan, [["SELL", "FERTILIZER", 100]])
        tiles = [t for row in self.obs.farms[0]["tiles"] for t in row
                 if isinstance(t, dict) and t.get("animal")]
        self.assertEqual(len(tiles), 12)
        self.assertTrue(all(t["consecutive_unfed"] == 0 for t in tiles))

    def advance_to(self, step):
        while self.obs.step < step:
            self.env.step([PASS, PASS])

    def test_fertilizer_pickup_application_and_water_lock_use_real_engine(self):
        self.advance_to(7 * 24)
        self.obs.farms[0]["tiles"][4][4] = self.engine._new_plant("TOMATO", 0, 24)
        self.obs.private["shed"]["FERTILIZER"] = 1
        self.obs.market["prices"].update(FERTILIZER=1, TOMATO=100)
        plan = CropPlan(None, None, 0, 0)
        actions, predicted, _ = self.execute(plan)
        self.assertEqual(actions[0], ["PICKUP", "FERTILIZER", 1])
        self.assertEqual(predicted["FERTILIZER"], 0)
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["FERTILIZE"])
        self.assertEqual(self.obs.farms[0]["tiles"][4][4]["fertilized_until_day"], 9)
        animal = self.engine._new_animal("COW", 0)
        animal.update(yield_units=6, fed_today=True)
        self.obs.farms[0]["tiles"][3][4] = animal
        actions, _, _ = self.execute(plan)
        self.assertEqual(actions[0], ["WATER"])
        self.assertTrue(self.obs.farms[0]["tiles"][4][4]["watered_today"])

    def test_parallel_fertilizer_pickups_respect_single_available_unit(self):
        self.advance_to(7 * 24)
        farm, private = self.obs.farms[0], self.obs.private
        farm["hands"] = [[5, 4]]
        private["inventories"] = [{}, {}]
        private["shed"]["FERTILIZER"] = 1
        for x in (4, 5):
            farm["tiles"][4][x] = self.engine._new_plant("TOMATO", 0, 24)
        self.obs.market["prices"].update(FERTILIZER=1, TOMATO=100)
        actions, predicted, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(sum(a[2] for a in actions if a[:2] == ["PICKUP", "FERTILIZER"]), 1)
        self.assertEqual(predicted["FERTILIZER"], 0)
        self.assertIn(["WATER"], actions)

    def test_full_crop_keeps_harvest_slot_instead_of_late_fertilizer(self):
        self.advance_to(8 * 24 + 22)
        tile = self.engine._new_plant("TOMATO", 0, 24)
        tile.update(yield_units=4)
        self.obs.farms[0]["tiles"][4][4] = tile
        self.obs.private["inventories"][0]["FERTILIZER"] = 1
        self.obs.market["prices"].update(FERTILIZER=1, TOMATO=100)
        actions, _, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["HARVEST"])

    def test_nightly_animal_cap_forces_collection_before_other_harvest(self):
        self.advance_to(9 * 24 + 22)
        farm = self.obs.farms[0]
        cow = self.engine._new_animal("COW", 0)
        cow.update(yield_units=5, pending_care_bonus=4, fed_today=True, cared_today=True)
        farm["tiles"][4][4] = cow
        melon = self.engine._new_plant("MELON", -1, 24)
        melon.update(yield_units=6, watered_today=True)
        farm["tiles"][3][4] = melon
        actions, _, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["HARVEST"])
        self.assertEqual(self.obs.private["inventories"][0]["MILK"], 5)
        self.env.step([PASS, PASS])
        self.assertEqual(self.obs.farms[0]["tiles"][4][4]["yield_units"], 5)

    def test_day_28_care_is_skipped_but_fertilizer_is_still_collected(self):
        self.advance_to(28 * 24)
        sheep = self.engine._new_animal("SHEEP", 11)
        sheep.update(fed_today=True, fertilizer_available=True, pending_care_bonus=2)
        self.obs.farms[0]["tiles"][4][4] = sheep
        actions, _, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["COLLECT_FERTILIZER"])
        self.assertFalse(self.obs.farms[0]["tiles"][4][4]["cared_today"])

    def test_late_courier_is_carrier_owned_and_predicts_deposit_for_sale(self):
        self.advance_to(22)
        farm, private = self.obs.farms[0], self.obs.private
        farm["farmer"] = [3, 4]
        private["inventories"][0]["MILK"] = 8
        private["shed"]["CARROT"] = 90
        actions, _, _ = self.execute(CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["EAST"])
        actions, predicted, _ = self.execute(CropPlan(None, None, 0, 0), [["SELL", "MILK", 8]])
        self.assertEqual(actions[0], ["DROP"])
        self.assertEqual(predicted["MILK"], 8)
        self.assertEqual(self.obs.private["shed"].get("MILK", 0), 0)


if __name__ == "__main__":
    unittest.main()
