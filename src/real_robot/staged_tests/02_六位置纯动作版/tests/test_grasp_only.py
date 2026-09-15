from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasp_only_test import (
    EXPECTED_SEQUENCE,
    execute_position,
    validate_config,
)


class FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def recenter_arm(self) -> None:
        self.calls.append(("recenter",))

    def run_route(self, route, label: str) -> None:
        self.calls.append(("route", route, label))

    def open_gripper(self) -> None:
        self.calls.append(("open",))

    def close_gripper(self) -> None:
        self.calls.append(("close",))

    def move_arm(self, point, label: str) -> None:
        self.calls.append(("arm", point, label))


class FakeEventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event: str, **details) -> None:
        self.events.append((event, details))


def calibrated_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["calibrated"] = True
    cfg["grasp_test"]["pickup_arm_profile"] = {
        "arm_pregrasp_mm": [10, 20],
        "arm_grasp_mm": [30, 40],
        "arm_lift_mm": [50, 60],
    }
    for destination in cfg["destinations"].values():
        destination.update(
            {
                "arm_predrop_mm": [70, 80],
                "arm_drop_mm": [90, 100],
                "arm_retract_mm": [110, 120],
            }
        )
    return cfg


class GraspOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "grasp_only_config.json").read_text(encoding="utf-8")
        )

    def test_dry_config_is_valid(self) -> None:
        validate_config(self.config, require_calibrated=False)

    def test_real_motion_is_locked_before_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_config(self.config, require_calibrated=True)

    def test_filled_test_config_is_valid_for_execution(self) -> None:
        validate_config(calibrated_config(self.config), require_calibrated=True)

    def test_sequence_matches_requested_destinations(self) -> None:
        actual = [
            (step["grid_id"], step["destination"])
            for step in self.config["grasp_test"]["fixed_sequence"]
        ]
        self.assertEqual(actual, EXPECTED_SEQUENCE)

    def test_all_routes_have_zero_commanded_net_change(self) -> None:
        for grid in self.config["grids"]:
            steps = grid["chassis_to_pick"] + grid["chassis_home_from_pick"]
            for key in ("x_m", "y_m", "z_deg"):
                self.assertAlmostEqual(sum(float(step.get(key, 0)) for step in steps), 0)
        for destination in self.config["destinations"].values():
            steps = (
                destination["chassis_to_destination"]
                + destination["chassis_home_from_destination"]
            )
            for key in ("x_m", "y_m", "z_deg"):
                self.assertAlmostEqual(sum(float(step.get(key, 0)) for step in steps), 0)

    def test_complete_six_position_call_counts(self) -> None:
        cfg = calibrated_config(self.config)
        controller = FakeController()
        event_log = FakeEventLog()
        grids = {grid["id"]: grid for grid in cfg["grids"]}
        for step in cfg["grasp_test"]["fixed_sequence"]:
            execute_position(
                controller,
                grids[step["grid_id"]],
                step["destination"],
                cfg,
                event_log,
            )
        names = [call[0] for call in controller.calls]
        self.assertEqual(names.count("close"), 6)
        self.assertEqual(names.count("open"), 12)
        self.assertEqual(names.count("arm"), 36)
        self.assertEqual(names.count("recenter"), 18)
        self.assertEqual(names.count("route"), 24)
        self.assertEqual(
            sum(name == "position_completed" for name, _ in event_log.events), 6
        )

    def test_entrypoint_has_no_vision_imports(self) -> None:
        source = (ROOT / "grasp_only_test.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"cv2", "torch", "ultralytics"}.isdisjoint(imported))
        self.assertNotIn("Detector", source)
        self.assertNotIn("scan_scene", source)


if __name__ == "__main__":
    unittest.main()
