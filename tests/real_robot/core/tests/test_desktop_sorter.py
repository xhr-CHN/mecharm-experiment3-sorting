from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_sorter import (
    Detection,
    GraspFailed,
    GridObservation,
    MotionError,
    RoboMasterController,
    apply_shape_guard,
    compute_home_correction,
    plan_candidates,
    point_to_grid,
    resolve_sdk_connection_types,
    run_task,
    sort_one,
    summarize_grid,
    validate_ap_local_ip,
    validate_config,
)


def detection(class_name: str, confidence: float = 0.9) -> Detection:
    class_ids = {"cup": 0, "mouse": 1, "bottle": 2}
    return Detection(class_ids[class_name], class_name, confidence, (0, 0, 10, 10), (0.2, 0.2))


def observation(
    grid_id: str,
    state: str,
    class_name: str | None = None,
    confidence: float = 0.0,
) -> GridObservation:
    return GridObservation(grid_id, state, class_name, confidence, 6, 1.0)


class RecordingController:
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

    def verify_grasp(self) -> None:
        self.calls.append(("verify",))

    def move_arm(self, point, label: str) -> None:
        self.calls.append(("arm", point, label))

    def start_camera_drain(self) -> None:
        pass

    def stop_camera_drain(self) -> None:
        pass


class RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event: str, **details) -> None:
        self.events.append((event, details))


class SorterLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "sorter_config.json").read_text(encoding="utf-8"))

    def test_config_is_valid_for_dry_run(self) -> None:
        validate_config(self.config, require_calibrated=False)
        self.assertEqual(self.config["model"]["confidence"], 0.45)

    def test_shape_guard_relabels_short_wide_bottle_prediction_as_cup(self) -> None:
        label, ratio, source = apply_shape_guard(
            "bottle",
            (0.0, 0.0, 100.0, 150.0),
            self.config["model"]["shape_guard"],
        )
        self.assertEqual(label, "cup")
        self.assertAlmostEqual(ratio, 1.5)
        self.assertEqual(source, "shape_guard")

    def test_shape_guard_keeps_tall_narrow_bottle(self) -> None:
        label, ratio, source = apply_shape_guard(
            "bottle",
            (0.0, 0.0, 100.0, 260.0),
            self.config["model"]["shape_guard"],
        )
        self.assertEqual(label, "bottle")
        self.assertAlmostEqual(ratio, 2.6)
        self.assertEqual(source, "shape_guard")

    def test_shape_guard_rejects_borderline_shape(self) -> None:
        label, ratio, source = apply_shape_guard(
            "cup",
            (0.0, 0.0, 100.0, 195.0),
            self.config["model"]["shape_guard"],
        )
        self.assertEqual(label, "cup_bottle_ambiguous")
        self.assertAlmostEqual(ratio, 1.95)
        self.assertEqual(source, "shape_guard")

    def test_shape_guard_does_not_change_mouse(self) -> None:
        label, ratio, source = apply_shape_guard(
            "mouse",
            (0.0, 0.0, 100.0, 50.0),
            self.config["model"]["shape_guard"],
        )
        self.assertEqual(label, "mouse")
        self.assertIsNone(ratio)
        self.assertEqual(source, "model")

    def test_config_rejects_overlapping_shape_thresholds(self) -> None:
        config = copy.deepcopy(self.config)
        config["model"]["shape_guard"]["bottle_min_aspect_ratio"] = 1.5
        with self.assertRaisesRegex(ValueError, "shape_guard"):
            validate_config(config, require_calibrated=False)

    def test_uncalibrated_config_blocks_execution(self) -> None:
        config = copy.deepcopy(self.config)
        config["calibrated"] = False
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_config(config, require_calibrated=True)

    def test_point_maps_to_exact_grid(self) -> None:
        self.assertEqual(point_to_grid((0.2, 0.7), self.config["grids"]), "G1")
        self.assertEqual(point_to_grid((0.25, 0.2), self.config["grids"]), "G4")
        self.assertIsNone(point_to_grid((0.34, 0.525), self.config["grids"]))

    def test_stable_cup_is_candidate(self) -> None:
        observation = summarize_grid(
            "G1",
            [detection("cup") for _ in range(5)] + [None],
            {"cup", "bottle"},
            min_hits=5,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "candidate")
        self.assertEqual(observation.class_name, "cup")

    def test_mouse_is_logged_as_unsupported(self) -> None:
        observation = summarize_grid(
            "G2",
            [detection("mouse") for _ in range(6)],
            {"cup", "bottle"},
            min_hits=5,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "unsupported")

    def test_conflicting_votes_are_ambiguous(self) -> None:
        observation = summarize_grid(
            "G3",
            [detection("cup") for _ in range(3)] + [detection("bottle") for _ in range(3)],
            {"cup", "bottle"},
            min_hits=3,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "ambiguous")

    def test_execution_requires_arm_points(self) -> None:
        config = copy.deepcopy(self.config)
        config["calibrated"] = True
        config["arm_point_profile"]["verified_on_this_robot"] = True
        config["arm_profiles"]["cup"]["arm_pregrasp_mm"] = None
        with self.assertRaisesRegex(ValueError, "arm_pregrasp_mm"):
            validate_config(config, require_calibrated=True)

    def test_profile_contains_all_twelve_class_specific_points(self) -> None:
        cup_pickup_expected = {
            "arm_pregrasp_mm": [0, -60],
            "arm_grasp_mm": [205, -60],
            "arm_lift_mm": [180, 100],
        }
        bottle_pickup_expected = {
            "arm_pregrasp_mm": [0, -15],
            "arm_grasp_mm": [205, -15],
            "arm_lift_mm": [180, 100],
        }
        drop_expected_by_class = {
            "cup": {
                "arm_predrop_mm": [180, 100],
                "arm_drop_mm": [205, -15],
                "arm_retract_mm": [150, 110],
            },
            "bottle": {
                "arm_predrop_mm": [180, 100],
                "arm_drop_mm": [205, -10],
                "arm_retract_mm": [150, 110],
            },
        }
        pickup_points = []
        drop_points = []
        self.assertFalse(self.config["arm_point_profile"]["shared_by_classes"])
        for class_name in ("cup", "bottle"):
            pickup_expected = (
                cup_pickup_expected if class_name == "cup" else bottle_pickup_expected
            )
            self.assertEqual(self.config["arm_profiles"][class_name], pickup_expected)
            pickup_points.extend(self.config["arm_profiles"][class_name].values())
            drop_expected = drop_expected_by_class[class_name]
            drop_points.extend(
                self.config["bins"][class_name][key]
                for key in drop_expected
            )
            self.assertEqual(
                {key: self.config["bins"][class_name][key] for key in drop_expected},
                drop_expected,
            )
        self.assertEqual(len(pickup_points) + len(drop_points), 12)
        self.assertTrue(all(point is not None for point in pickup_points + drop_points))
        self.assertEqual(self.config["model"]["imgsz"], 800)

    def test_filled_points_stay_blocked_until_physically_verified(self) -> None:
        config = copy.deepcopy(self.config)
        config["calibrated"] = True
        config["arm_point_profile"]["verified_on_this_robot"] = False
        with self.assertRaisesRegex(ValueError, "verified_on_this_robot"):
            validate_config(config, require_calibrated=True)

    def test_verified_and_calibrated_nominal_profile_unlocks_validation(self) -> None:
        config = copy.deepcopy(self.config)
        config["calibrated"] = True
        config["arm_point_profile"]["verified_on_this_robot"] = True
        validate_config(config, require_calibrated=True)

    def test_shared_profile_rejects_divergent_class_points(self) -> None:
        config = copy.deepcopy(self.config)
        config["arm_point_profile"]["shared_by_classes"] = True
        config["arm_profiles"]["bottle"]["arm_grasp_mm"] = [190, -10]
        with self.assertRaisesRegex(ValueError, "pickup points differ"):
            validate_config(config, require_calibrated=False)

    def test_front_row_has_priority_over_higher_confidence_rear_row(self) -> None:
        observations = [
            observation("G1", "candidate", "cup", 0.60),
            observation("G2", "empty"),
            observation("G3", "candidate", "bottle", 0.70),
            observation("G4", "candidate", "bottle", 0.99),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual([item.grid_id for item in candidates], ["G1", "G3"])
        self.assertEqual(blocked[0]["grid_id"], "G4")

    def test_rear_grid_is_released_after_front_lane_is_empty(self) -> None:
        observations = [
            observation("G1", "empty"),
            observation("G2", "empty"),
            observation("G3", "empty"),
            observation("G4", "candidate", "cup", 0.90),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual([item.grid_id for item in candidates], ["G4"])
        self.assertEqual(blocked, [])

    def test_mouse_in_front_lane_blocks_rear_drive_path(self) -> None:
        observations = [
            observation("G1", "unsupported", "mouse", 0.95),
            observation("G2", "empty"),
            observation("G3", "empty"),
            observation("G4", "candidate", "bottle", 0.90),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual(candidates, [])
        self.assertEqual(
            blocked,
            [{"grid_id": "G4", "blocked_by": "G1", "blocker_state": "unsupported"}],
        )

    def test_bin_turn_directions_match_left_and_right(self) -> None:
        self.assertEqual(self.config["bins"]["cup"]["chassis_to_bin"], [{"z_deg": 90.0}])
        self.assertEqual(
            self.config["bins"]["cup"]["chassis_home_from_bin"],
            [{"z_deg": -90.0}],
        )
        self.assertEqual(
            self.config["bins"]["bottle"]["chassis_to_bin"],
            [{"z_deg": -90.0}],
        )
        self.assertEqual(
            self.config["bins"]["bottle"]["chassis_home_from_bin"],
            [{"z_deg": 90.0}],
        )

    def test_bin_centers_are_reachable_after_in_place_turn(self) -> None:
        reach = self.config["layout"]["nominal_arm_reach_m"]
        self.assertEqual(
            self.config["layout"]["cup_zone_center_m"],
            {"x_forward_m": 0.0, "y_right_m": -reach},
        )
        self.assertEqual(
            self.config["layout"]["bottle_zone_center_m"],
            {"x_forward_m": 0.0, "y_right_m": reach},
        )

    def test_row_edge_gap_is_seven_centimeters(self) -> None:
        front_x = self.config["grids"][0]["ground_center_m"]["x_forward_m"]
        rear_x = self.config["grids"][3]["ground_center_m"]["x_forward_m"]
        square_size = self.config["layout"]["grid_square_size_m"]
        self.assertAlmostEqual(rear_x - front_x - square_size, 0.07)

    def test_rear_rois_are_shifted_down_without_front_overlap(self) -> None:
        for rear_grid in self.config["grids"][3:]:
            self.assertAlmostEqual(rear_grid["roi_norm"][1], 0.15)
            self.assertAlmostEqual(rear_grid["roi_norm"][3], 0.50)
        for front_grid, rear_grid in zip(self.config["grids"][:3], self.config["grids"][3:]):
            self.assertAlmostEqual(front_grid["roi_norm"][1] - rear_grid["roi_norm"][3], 0.05)

    def test_pick_routes_use_field_calibrated_advances(self) -> None:
        self.assertAlmostEqual(self.config["layout"]["nominal_arm_reach_m"], 0.40)
        for grid in self.config["grids"][:3]:
            forward = sum(step.get("x_m", 0.0) for step in grid["motion"]["chassis_to_pick"])
            self.assertAlmostEqual(forward, 0.06)
        for grid in self.config["grids"][3:]:
            forward = sum(step.get("x_m", 0.0) for step in grid["motion"]["chassis_to_pick"])
            self.assertAlmostEqual(forward, 0.48)

    def test_chassis_position_subscription_captures_initial_odometry(self) -> None:
        def subscribe(*, cs: int, freq: int, callback: object) -> bool:
            self.assertEqual(cs, 0)
            self.assertEqual(freq, 20)
            callback((0.01, -0.02, 0.5))
            return True

        controller = RoboMasterController.__new__(RoboMasterController)
        controller.cfg = copy.deepcopy(self.config)
        controller.cfg["safety"]["route_test_home_correction"]["position_warmup_s"] = 0
        controller.chassis = SimpleNamespace(sub_position=subscribe)
        controller.chassis_position = None
        controller._chassis_position_subscribed = False
        position = controller.enable_chassis_position()

        self.assertEqual(position, (0.01, -0.02, 0.5))
        self.assertTrue(controller._chassis_position_subscribed)

    def test_home_correction_uses_gain_tolerance_and_cap(self) -> None:
        correction_cfg = self.config["safety"]["route_test_home_correction"]
        self.assertEqual(
            compute_home_correction((0.08, -0.004, 0.0), correction_cfg),
            (-0.048, 0.0),
        )
        self.assertEqual(
            compute_home_correction((0.15, 0.0, 0.0), correction_cfg),
            (-0.05, 0.0),
        )
        self.assertIsNone(compute_home_correction((0.01, 0.0, 0.0), correction_cfg))

    def test_home_correction_rejects_implausible_odometry(self) -> None:
        correction_cfg = self.config["safety"]["route_test_home_correction"]
        with self.assertRaisesRegex(MotionError, "implausible"):
            compute_home_correction((0.6, 0.0, 0.0), correction_cfg)

    def test_json_connection_values_resolve_to_sdk_constant_objects(self) -> None:
        fake_conn = SimpleNamespace(
            CONNECTION_WIFI_AP=object(),
            CONNECTION_WIFI_STA=object(),
            CONNECTION_USB_RNDIS=object(),
            CONNECTION_PROTO_UDP=object(),
            CONNECTION_PROTO_TCP=object(),
        )
        conn_type, proto_type = resolve_sdk_connection_types(
            self.config["robot"], fake_conn
        )
        self.assertIs(conn_type, fake_conn.CONNECTION_WIFI_AP)
        self.assertIs(proto_type, fake_conn.CONNECTION_PROTO_UDP)

    def test_ap_mode_rejects_non_robot_network(self) -> None:
        self.assertEqual(validate_ap_local_ip("192.168.2.20"), "192.168.2.20")
        for bad_ip in ("10.0.0.8", "192.168.2.1", "not-an-ip"):
            with self.subTest(bad_ip=bad_ip), self.assertRaises(MotionError):
                validate_ap_local_ip(bad_ip)

    def test_every_chassis_home_route_is_exact_inverse(self) -> None:
        route_pairs = []
        for grid in self.config["grids"]:
            route_pairs.append(
                (
                    grid["motion"]["chassis_to_pick"],
                    grid["motion"]["chassis_home_from_pick"],
                )
            )
        for destination in self.config["bins"].values():
            route_pairs.append(
                (
                    destination["chassis_to_bin"],
                    destination["chassis_home_from_bin"],
                )
            )

        for outbound, home in route_pairs:
            expected_home = [
                {key: -value for key, value in step.items()}
                for step in reversed(outbound)
            ]
            self.assertEqual(home, expected_home)

    def test_run_route_waits_for_settle_after_every_chassis_step(self) -> None:
        class SuccessfulAction:
            has_succeeded = True
            failure_reason = None

            @staticmethod
            def wait_for_completed(timeout: float) -> bool:
                return True

        move_calls = []

        def move(**kwargs):
            move_calls.append(kwargs)
            return SuccessfulAction()

        controller = RoboMasterController.__new__(RoboMasterController)
        controller.cfg = copy.deepcopy(self.config)
        controller.chassis = SimpleNamespace(move=move)
        route = [{"x_m": -0.12}, {"y_m": 0.4}]

        with patch("desktop_sorter.time.sleep") as sleep_mock:
            controller.run_route(route, "test return G1")

        self.assertEqual(len(move_calls), 2)
        self.assertEqual(
            sleep_mock.call_args_list,
            [
                unittest.mock.call(2.0),
                unittest.mock.call(2.0),
            ],
        )

    def test_sort_one_uses_detected_class_profile_and_destination(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["arm_profiles"]["bottle"] = {
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
        controller = RecordingController()
        event_log = RecordingLog()

        sort_one(controller, cfg["grids"][1], "bottle", cfg, event_log)

        self.assertEqual(controller.calls[1][1], cfg["grids"][1]["motion"]["chassis_to_pick"])
        self.assertEqual(controller.calls[3][1], [1, 2])
        self.assertEqual(controller.calls[4][1], [3, 4])
        self.assertEqual(controller.calls[7][1], [5, 6])
        self.assertEqual(controller.calls[10][1], cfg["bins"]["bottle"]["chassis_to_bin"])
        self.assertEqual(controller.calls[11][1], [7, 8])
        self.assertEqual(controller.calls[12][1], [9, 10])
        self.assertEqual(controller.calls[14][1], [11, 12])
        self.assertEqual(event_log.events[-1][0], "sort_completed")

    def test_close_gripper_uses_final_update_and_accepts_robot_success_states(self) -> None:
        controller = RoboMasterController.__new__(RoboMasterController)
        controller.cfg = copy.deepcopy(self.config)
        controller.cfg["safety"]["gripper_close_seconds"] = 0
        controller.cfg["safety"]["gripper_status_timeout_s"] = 0.01
        controller.gripper_status = "closed"

        class FakeGripper:
            def close(self, power: int) -> bool:
                return True

            def pause(inner_self) -> bool:
                controller.gripper_status = "normal"
                return True

        controller.gripper = FakeGripper()
        controller.close_gripper()
        controller.verify_grasp()
        controller.gripper_status = "closed"
        controller.verify_grasp()
        controller.gripper_status = "opened"
        with self.assertRaises(GraspFailed):
            controller.verify_grasp()

    @patch("desktop_sorter.time.sleep", return_value=None)
    def test_camera_read_retries_transient_empty_queue(self, _sleep: object) -> None:
        controller = RoboMasterController.__new__(RoboMasterController)

        class FakeCamera:
            def __init__(self) -> None:
                self.calls = 0

            def read_cv2_image(self, timeout: int, strategy: str) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise Empty
                return "frame"

        controller.camera = FakeCamera()
        self.assertEqual(controller.read_frame(), "frame")
        self.assertEqual(controller.camera.calls, 3)

    @patch("desktop_sorter.time.sleep", return_value=None)
    def test_camera_read_reports_persistent_empty_queue(self, _sleep: object) -> None:
        controller = RoboMasterController.__new__(RoboMasterController)

        class FakeCamera:
            def read_cv2_image(self, timeout: int, strategy: str) -> None:
                raise Empty

        controller.camera = FakeCamera()
        with self.assertRaisesRegex(MotionError, "no frame after 3 attempts"):
            controller.read_frame()

    def test_camera_drain_consumes_frames_during_motion(self) -> None:
        controller = RoboMasterController.__new__(RoboMasterController)
        controller._camera_drain_stop = threading.Event()
        controller._camera_drain_thread = None
        frame_seen = threading.Event()

        class FakeCamera:
            def read_cv2_image(self, timeout: int, strategy: str) -> object:
                frame_seen.set()
                controller._camera_drain_stop.wait(0.01)
                return object()

        controller.camera = FakeCamera()
        controller.start_camera_drain()
        self.assertTrue(frame_seen.wait(1.0))
        controller.stop_camera_drain()
        self.assertIsNone(controller._camera_drain_thread)

    def test_full_state_machine_sorts_one_cup_and_verifies_source_empty(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["camera"]["scan_settle_s"] = 0
        cfg["arm_profiles"]["cup"] = {
            "arm_pregrasp_mm": [1, 2],
            "arm_grasp_mm": [3, 4],
            "arm_lift_mm": [5, 6],
        }
        cfg["bins"]["cup"].update(
            {
                "arm_predrop_mm": [7, 8],
                "arm_drop_mm": [9, 10],
                "arm_retract_mm": [11, 12],
            }
        )
        initial = [
            observation("G1", "candidate", "cup", 0.9),
            *[observation(f"G{index}", "empty") for index in range(2, 7)],
        ]
        cleared = [observation(f"G{index}", "empty") for index in range(1, 7)]
        controller = RecordingController()
        event_log = RecordingLog()

        with tempfile.TemporaryDirectory() as temporary_directory:
            event_log.run_dir = Path(temporary_directory)
            with patch("desktop_sorter.scan_scene", side_effect=[initial, cleared]) as scan:
                summary = run_task(
                    detector=object(),
                    read_frame=lambda: None,
                    controller=controller,
                    cfg=cfg,
                    event_log=event_log,
                    execute=True,
                    show=False,
                    max_objects=1,
                )

        self.assertEqual(summary["sorted_total"], 1)
        self.assertEqual(summary["sorted_by_class"], {"cup": 1})
        self.assertEqual(scan.call_count, 2)
        event_names = [name for name, _ in event_log.events]
        self.assertIn("source_grid_cleared", event_names)
        self.assertEqual(event_names.count("scan_pose_ready"), 2)
        self.assertEqual(controller.calls.count(("recenter",)), 5)

    def test_execution_recenters_arm_before_the_first_scan(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["camera"]["scan_settle_s"] = 0
        cfg["task"]["stop_after_no_candidate_scans"] = 1
        controller = RecordingController()
        event_log = RecordingLog()
        empty_scene = [observation(f"G{index}", "empty") for index in range(1, 7)]

        def verify_pose_before_scan(*_args, **_kwargs):
            self.assertEqual(controller.calls, [("recenter",)])
            return empty_scene

        with tempfile.TemporaryDirectory() as temporary_directory:
            event_log.run_dir = Path(temporary_directory)
            with patch("desktop_sorter.scan_scene", side_effect=verify_pose_before_scan):
                run_task(
                    detector=object(),
                    read_frame=lambda: None,
                    controller=controller,
                    cfg=cfg,
                    event_log=event_log,
                    execute=True,
                    show=False,
                    max_objects=1,
                )

        self.assertIn("scan_pose_ready", [name for name, _ in event_log.events])

    def test_preview_does_not_move_arm_and_warns_about_manual_recenter(self) -> None:
        cfg = copy.deepcopy(self.config)
        controller = RecordingController()
        event_log = RecordingLog()
        empty_scene = [observation(f"G{index}", "empty") for index in range(1, 7)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            event_log.run_dir = Path(temporary_directory)
            with patch("desktop_sorter.scan_scene", return_value=empty_scene):
                run_task(
                    detector=object(),
                    read_frame=lambda: None,
                    controller=controller,
                    cfg=cfg,
                    event_log=event_log,
                    execute=False,
                    show=False,
                    max_objects=1,
                )

        self.assertEqual(controller.calls, [])
        self.assertIn("scan_pose_not_enforced", [name for name, _ in event_log.events])

    def test_recenter_preview_moves_only_to_scan_pose(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["camera"]["scan_settle_s"] = 0
        controller = RecordingController()
        event_log = RecordingLog()
        empty_scene = [observation(f"G{index}", "empty") for index in range(1, 7)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            event_log.run_dir = Path(temporary_directory)
            with patch("desktop_sorter.scan_scene", return_value=empty_scene):
                summary = run_task(
                    detector=object(),
                    read_frame=lambda: None,
                    controller=controller,
                    cfg=cfg,
                    event_log=event_log,
                    execute=False,
                    show=False,
                    max_objects=1,
                    enforce_scan_pose=True,
                )

        self.assertEqual(controller.calls, [("recenter",)])
        self.assertFalse(summary["execute"])
        event_names = [name for name, _ in event_log.events]
        self.assertIn("scan_pose_ready", event_names)
        self.assertIn("dry_run_complete", event_names)


if __name__ == "__main__":
    unittest.main()
