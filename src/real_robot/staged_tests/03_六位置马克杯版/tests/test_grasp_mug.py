from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasp_mug_test import (
    EXPECTED_SEQUENCE,
    MotionError,
    execute_position,
    resolve_sdk_connection_types,
    select_sequence,
    validate_ap_local_ip,
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
    cfg = json.loads(json.dumps(config))
    cfg["calibrated"] = True
    return cfg


class GraspMugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "grasp_mug_config.json").read_text(encoding="utf-8")
        )

    def test_dry_config_is_valid(self) -> None:
        validate_config(self.config)

    def test_real_motion_is_locked_before_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_config(
                self.config,
                require_arm_points=True,
                require_calibrated=True,
            )

    def test_nominal_points_are_complete_for_acknowledged_execution(self) -> None:
        validate_config(
            self.config,
            require_arm_points=True,
            require_calibrated=False,
        )

    def test_calibrated_config_is_valid_for_execution(self) -> None:
        validate_config(
            calibrated_config(self.config),
            require_arm_points=True,
            require_calibrated=True,
        )

    def test_nominal_mug_points_match_reviewed_profile(self) -> None:
        pickup = self.config["grasp_test"]["pickup_arm_profile"]
        self.assertEqual(pickup["arm_pregrasp_mm"], [180, 100])
        self.assertEqual(pickup["arm_grasp_mm"], [205, -15])
        self.assertEqual(pickup["arm_lift_mm"], [180, 100])
        for destination in self.config["destinations"].values():
            self.assertEqual(destination["arm_predrop_mm"], [180, 100])
            self.assertEqual(destination["arm_drop_mm"], [205, -10])
            self.assertEqual(destination["arm_retract_mm"], [150, 110])

    def test_single_grid_selection(self) -> None:
        sequence = self.config["grasp_test"]["fixed_sequence"]
        self.assertEqual(
            select_sequence(sequence, "G1"),
            [{"grid_id": "G1", "destination": "right"}],
        )
        self.assertEqual(select_sequence(sequence, None), sequence)

    def test_json_connection_values_resolve_to_sdk_constants(self) -> None:
        from robomaster import conn

        conn_type, proto_type = resolve_sdk_connection_types(
            self.config["robot"], conn
        )
        self.assertIs(conn_type, conn.CONNECTION_WIFI_AP)
        self.assertIs(proto_type, conn.CONNECTION_PROTO_UDP)

    def test_ap_local_ip_validation(self) -> None:
        self.assertEqual(validate_ap_local_ip("192.168.2.20"), "192.168.2.20")
        for bad_ip in ("10.140.245.187", "192.168.2.1", "not-an-ip"):
            with self.subTest(bad_ip=bad_ip), self.assertRaises(MotionError):
                validate_ap_local_ip(bad_ip)

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
        source = (ROOT / "grasp_mug_test.py").read_text(encoding="utf-8")
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
