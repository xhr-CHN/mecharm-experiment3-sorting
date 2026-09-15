from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed_sequence_grasp import (
    EXPECTED_SEQUENCE,
    RoboMasterController,
    execute_position,
    validate_config,
)


class FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def restore_camera_home(self, label: str) -> None:
        self.calls.append(("camera_home", label))

    def recenter_arm(self, label: str = "arm recenter") -> None:
        self.calls.append(("recenter", label))

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


class FakeChassis:
    def __init__(self) -> None:
        self.commands: list[dict] = []

    def drive_speed(self, **kwargs):
        self.commands.append(kwargs)
        return True


class FixedSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "fixed_sequence_config.json").read_text(encoding="utf-8")
        )

    def test_config_is_valid_for_execution(self) -> None:
        validate_config(self.config, require_calibrated=True)

    def test_sequence_is_exactly_requested(self) -> None:
        actual = [
            (step["grid_id"], step["destination"])
            for step in self.config["task"]["fixed_sequence"]
        ]
        self.assertEqual(actual, EXPECTED_SEQUENCE)

    def test_only_g2_goes_left(self) -> None:
        left = [
            step["grid_id"]
            for step in self.config["task"]["fixed_sequence"]
            if step["destination"] == "left"
        ]
        self.assertEqual(left, ["G2"])

    def test_all_routes_return_to_commanded_origin(self) -> None:
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

    def test_chassis_speeds_are_lower_than_original(self) -> None:
        self.assertEqual(self.config["safety"]["translation_speed_mps"], 0.35)
        self.assertLess(self.config["safety"]["translation_speed_mps"], 0.5)
        self.assertEqual(self.config["safety"]["default_z_speed_dps"], 20.0)
        self.assertLess(self.config["safety"]["default_z_speed_dps"], 30.0)

    def test_low_speed_translation_always_ends_with_stop(self) -> None:
        controller = RoboMasterController.__new__(RoboMasterController)
        controller.chassis = FakeChassis()
        with patch("fixed_sequence_grasp.time.sleep") as sleep:
            controller.drive_translation(0.35, 0.0, 0.35, "test")
        self.assertAlmostEqual(controller.chassis.commands[0]["x"], 0.35)
        self.assertEqual(controller.chassis.commands[0]["y"], 0.0)
        self.assertEqual(
            controller.chassis.commands[-1], {"x": 0, "y": 0, "z": 0, "timeout": 1}
        )
        sleep.assert_called_once_with(1.0)

    def test_all_six_positions_keep_camera_home_and_pause_flow(self) -> None:
        controller = FakeController()
        event_log = FakeEventLog()
        grids = {grid["id"]: grid for grid in self.config["grids"]}
        for step in self.config["task"]["fixed_sequence"]:
            execute_position(
                controller,
                grids[step["grid_id"]],
                step["destination"],
                self.config,
                event_log,
            )
        names = [call[0] for call in controller.calls]
        self.assertEqual(names.count("close"), 6)
        self.assertEqual(names.count("open"), 12)
        self.assertEqual(names.count("camera_home"), 12)
        self.assertEqual(names.count("recenter"), 12)
        self.assertEqual(
            [details["grid_id"] for event, details in event_log.events if event == "position_completed"],
            ["G2", "G5", "G1", "G4", "G3", "G6"],
        )

    def test_g2_uses_left_profile_and_g5_uses_right_profile(self) -> None:
        grids = {grid["id"]: grid for grid in self.config["grids"]}
        event_log = FakeEventLog()
        g2_controller = FakeController()
        execute_position(g2_controller, grids["G2"], "left", self.config, event_log)
        g5_controller = FakeController()
        execute_position(g5_controller, grids["G5"], "right", self.config, event_log)
        g2_arm_points = [call[1] for call in g2_controller.calls if call[0] == "arm"]
        g5_arm_points = [call[1] for call in g5_controller.calls if call[0] == "arm"]
        self.assertEqual(g2_arm_points[0], [0, -60])
        self.assertEqual(g5_arm_points[0], [0, -15])

    def test_source_has_no_vision_or_recognition_dependencies(self) -> None:
        source = (ROOT / "fixed_sequence_grasp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue({"cv2", "torch", "ultralytics"}.isdisjoint(imported))
        for forbidden in ("Detector", "scan_scene", "read_frame", "sub_status", "verify_grasp"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
