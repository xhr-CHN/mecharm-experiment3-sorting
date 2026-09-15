"""Fixed six-grid RoboMaster grasp-motion test with YOLO observations.

Recognition is retained for logging, but it never gates motion in this test:
G1-G3 are always grasped toward the right bin and G4-G6 toward the left bin.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNTIME_CACHE = ROOT / ".runtime_cache"
(RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_CACHE / "ultralytics"))

from desktop_sorter import (
    ConfigError,
    Detector,
    EventLog,
    RoboMasterController,
    UsbCamera,
    _validate_arm_point,
    load_config,
    scan_scene,
    validate_config,
)


DEFAULT_CONFIG = ROOT / "grasp_test_config.json"
EXPECTED_SEQUENCE = [
    ("G1", "right"),
    ("G2", "right"),
    ("G3", "right"),
    ("G4", "left"),
    ("G5", "left"),
    ("G6", "left"),
]


def validate_grasp_test_config(
    cfg: dict[str, Any], require_calibrated: bool = False
) -> None:
    """Validate the base detector/routes and test-only arm points."""
    validate_config(cfg, require_calibrated=False)
    test_cfg = cfg.get("grasp_test")
    if not isinstance(test_cfg, dict):
        raise ConfigError("Missing grasp_test section")
    if test_cfg.get("recognition_controls_motion") is not False:
        raise ConfigError("grasp_test.recognition_controls_motion must be false")
    if test_cfg.get("verify_grasp") is not False:
        raise ConfigError("grasp_test.verify_grasp must be false for empty-grid testing")

    sequence = test_cfg.get("fixed_sequence")
    actual_sequence = [
        (step.get("grid_id"), step.get("destination"))
        for step in sequence
    ] if isinstance(sequence, list) else []
    if actual_sequence != EXPECTED_SEQUENCE:
        raise ConfigError(
            "grasp_test.fixed_sequence must send G1-G3 right and G4-G6 left"
        )
    if test_cfg.get("destination_bins") != {"left": "cup", "right": "bottle"}:
        raise ConfigError("grasp_test.destination_bins must map left/right correctly")

    if not require_calibrated:
        return
    if cfg.get("calibrated") is not True:
        raise ConfigError(
            'Real test motion is blocked until grasp_test_config.json contains "calibrated": true'
        )
    pickup = test_cfg.get("pickup_arm_profile", {})
    for key in ("arm_pregrasp_mm", "arm_grasp_mm", "arm_lift_mm"):
        _validate_arm_point(pickup.get(key), f"grasp_test.pickup_arm_profile.{key}", cfg)
    for bin_name in ("cup", "bottle"):
        drop = cfg["bins"][bin_name]
        for key in ("arm_predrop_mm", "arm_drop_mm", "arm_retract_mm"):
            _validate_arm_point(drop.get(key), f"bins.{bin_name}.{key}", cfg)


def fixed_sequence(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Return a defensive copy of the six fixed motion assignments."""
    return [dict(step) for step in cfg["grasp_test"]["fixed_sequence"]]


def execute_grid_motion(
    controller: RoboMasterController,
    grid: dict[str, Any],
    destination: str,
    cfg: dict[str, Any],
    event_log: EventLog,
) -> None:
    """Run one complete grasp-and-place motion without object-state checks."""
    route = grid["motion"]
    pickup = cfg["grasp_test"]["pickup_arm_profile"]
    bin_name = cfg["grasp_test"]["destination_bins"][destination]
    drop = cfg["bins"][bin_name]
    event_log.write(
        "forced_grasp_motion_started",
        grid_id=grid["id"],
        destination=destination,
        recognition_used_for_motion=False,
        grasp_verification=False,
    )

    controller.recenter_arm()
    controller.run_route(route["chassis_to_pick"], f"test route to {grid['id']}")
    controller.open_gripper()
    controller.move_arm(pickup["arm_pregrasp_mm"], f"{grid['id']} test pregrasp")
    controller.move_arm(pickup["arm_grasp_mm"], f"{grid['id']} test grasp")
    controller.close_gripper()
    controller.move_arm(pickup["arm_lift_mm"], f"{grid['id']} test lift")
    controller.recenter_arm()
    controller.run_route(
        route["chassis_home_from_pick"], f"test return from {grid['id']}"
    )

    controller.run_route(drop["chassis_to_bin"], f"test route to {destination} bin")
    controller.move_arm(drop["arm_predrop_mm"], f"{destination} test predrop")
    controller.move_arm(drop["arm_drop_mm"], f"{destination} test drop")
    controller.open_gripper()
    controller.move_arm(drop["arm_retract_mm"], f"{destination} test retract")
    controller.recenter_arm()
    controller.run_route(
        drop["chassis_home_from_bin"], f"test return from {destination} bin"
    )
    event_log.write(
        "forced_grasp_motion_completed",
        grid_id=grid["id"],
        destination=destination,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--max-actions",
        type=int,
        default=6,
        help="Run only the first N fixed grid actions (1-6); default: 6",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs_grasp_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_actions <= len(EXPECTED_SEQUENCE):
        raise ConfigError("--max-actions must be an integer from 1 to 6")
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    validate_grasp_test_config(cfg, require_calibrated=args.execute)
    if args.validate_config:
        print(
            f"Grasp-test configuration is valid for "
            f"{'execution' if args.execute else 'dry-run'}: {config_path}"
        )
        return

    event_log = EventLog(args.log_dir.resolve())
    controller: RoboMasterController | None = None
    usb_camera: UsbCamera | None = None
    try:
        detector = Detector(cfg, config_path)
        if cfg["camera"]["source"] == "robomaster":
            controller = RoboMasterController(
                cfg, needs_camera=True, motion_enabled=args.execute
            )
            read_frame = controller.read_frame
        else:
            usb_camera = UsbCamera(cfg)
            read_frame = usb_camera.read_frame
            if args.execute:
                controller = RoboMasterController(
                    cfg, needs_camera=False, motion_enabled=True
                )

        grids = {grid["id"]: grid for grid in cfg["grids"]}
        sequence = fixed_sequence(cfg)[: args.max_actions]
        event_log.write(
            "grasp_motion_test_started",
            execute=args.execute,
            fixed_sequence=sequence,
            warning="recognition is logged but does not control motion",
        )

        if not args.execute:
            observations = scan_scene(
                detector, read_frame, cfg, event_log, 1, not args.no_display
            )
            event_log.write(
                "grasp_motion_test_dry_run",
                observations=[asdict(item) for item in observations],
                planned=sequence,
            )
            return

        assert controller is not None
        scan_index = 0
        for action_index, assignment in enumerate(sequence, start=1):
            scan_index += 1
            observations = scan_scene(
                detector,
                read_frame,
                cfg,
                event_log,
                scan_index,
                not args.no_display,
            )
            observed = next(
                item for item in observations if item.grid_id == assignment["grid_id"]
            )
            event_log.write(
                "recognition_observed_before_forced_motion",
                action_index=action_index,
                observation=asdict(observed),
                action_will_run_regardless=True,
            )
            execute_grid_motion(
                controller,
                grids[assignment["grid_id"]],
                assignment["destination"],
                cfg,
                event_log,
            )

        scan_index += 1
        final_observations = scan_scene(
            detector,
            read_frame,
            cfg,
            event_log,
            scan_index,
            not args.no_display,
        )
        event_log.write(
            "grasp_motion_test_completed",
            motions_completed=len(sequence),
            final_observations=[asdict(item) for item in final_observations],
        )
    except Exception as error:
        event_log.write("grasp_motion_test_exception", error=repr(error))
        if controller is not None:
            controller.stop_safely()
        raise
    finally:
        if usb_camera is not None:
            usb_camera.close()
        if controller is not None:
            controller.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        event_log.close()


if __name__ == "__main__":
    main()
