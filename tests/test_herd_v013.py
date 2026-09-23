"""The expanded herd must remain serviceable by the configured workforce."""
import unittest

from kaggriculture_agent.domain import distance, game_configuration, nearest_shed
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from kaggriculture_agent.scheduler import TaskScheduler
from scripts.benchmark import load_engine, make_environment


class ExpandedHerdTests(unittest.TestCase):
    def test_eight_cows_six_sheep_are_fed_and_survive_production(self):
        engine = load_engine()
        env = make_environment(9, 720, {"weedSpawnChance": 0})
        farm = env.state[0].observation.farms[0]
        farm["money"] = 10000
        engine._do_buy_land(farm, 10)
        engine._do_buy_land(farm, 10)
        positions = sorted(((x, y) for y in range(10) for x in range(10)
                            if farm["tiles"][y][x] != "LOCKED"),
                           key=lambda p: (distance(p, nearest_shed(p, 10)), p[1], p[0]))[:14]
        for i, (x, y) in enumerate(positions):
            farm["tiles"][y][x] = engine._new_animal("COW" if i < 8 else "SHEEP", 0)
        plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 8, "SHEEP": 6})
        scheduler, config = TaskScheduler(PolicyConfig()), game_configuration()
        for _ in range(8 * 24):
            obs = env.state[0].observation
            if obs.hour == 0:
                obs.private["shed"]["WHEAT"] = 14
                obs.farms[0]["hands"] = [[4, 4], [5, 4], [4, 5], [5, 5], [4, 4], [5, 4], [4, 5]]
                obs.private["inventories"] = [{} for _ in range(8)]
            animals = [t for row in obs.farms[0]["tiles"] for t in row
                       if isinstance(t, dict) and t.get("animal")]
            self.assertEqual(len(animals), 14)
            if obs.hour == 23:
                self.assertTrue(all(t["fed_today"] for t in animals))
            actions, _, _ = scheduler.actions(obs, config, plan)
            env.step([{"farmer": actions[0], "hands": actions[1:],
                       "market": [["SELL", product, 100] for product in ("MILK", "WOOL", "FERTILIZER")]},
                      {"farmer": ["PASS"]}])
        animals = [t for row in env.state[0].observation.farms[0]["tiles"] for t in row
                   if isinstance(t, dict) and t.get("animal")]
        self.assertEqual(len(animals), 14)
        self.assertTrue(all(t["consecutive_unfed"] == 0 for t in animals))


if __name__ == "__main__":
    unittest.main()
