from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasp_motion_test import (
    EXPECTED_SEQUENCE,
    execute_grid_motion,
    fixed_sequence,
    validate_grasp_test_config,
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

    def verify_grasp(self) -> None:
        raise AssertionError("empty-grid test must not verify grasp status")


class FakeEventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event: str, **details) -> None:
        self.events.append((event, details))


class GraspMotionTestLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "grasp_test_config.json").read_text(encoding="utf-8")
        )

    def test_dry_config_is_valid(self) -> None:
        validate_grasp_test_config(self.config, require_calibrated=False)

    def test_execution_is_locked_before_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_grasp_test_config(self.config, require_calibrated=True)

    def test_fixed_sequence_matches_requested_sides(self) -> None:
        actual = [
            (step["grid_id"], step["destination"])
            for step in fixed_sequence(self.config)
        ]
        self.assertEqual(actual, EXPECTED_SEQUENCE)

    def test_motion_runs_without_grasp_verification(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["grasp_test"]["pickup_arm_profile"] = {
            "arm_pregrasp_mm": [1, 2],
            "arm_grasp_mm": [3, 4],
            "arm_lift_mm": [5, 6],
        }
        cfg["bins"]["bottle"].update(
            {
                "arm_predrop_mm": [7, 8],
                "arm_drop_mm": [9, 10],
                "arm_retract_mm": [11, 12],
            }
        )
        controller = FakeController()
        event_log = FakeEventLog()

        execute_grid_motion(
            controller,
            cfg["grids"][0],
            "right",
            cfg,
            event_log,
        )

        call_names = [call[0] for call in controller.calls]
        self.assertEqual(call_names.count("close"), 1)
        self.assertEqual(call_names.count("open"), 2)
        self.assertNotIn("verify", call_names)
        self.assertEqual(
            controller.calls[1][1], cfg["grids"][0]["motion"]["chassis_to_pick"]
        )
        self.assertEqual(
            controller.calls[9][1], cfg["bins"]["bottle"]["chassis_to_bin"]
        )
        self.assertEqual(event_log.events[-1][0], "forced_grasp_motion_completed")


if __name__ == "__main__":
    unittest.main()
