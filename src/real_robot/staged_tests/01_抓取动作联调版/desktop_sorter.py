"""RoboMaster EP ground-grid cup/bottle sorting with a YOLO model.

The program is safe-by-default: without --execute it only scans and reports.
Real movement additionally requires "calibrated": true in sorter_config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "sorter_config.json"


class ConfigError(ValueError):
    """The sorter configuration is incomplete or unsafe."""


class MotionError(RuntimeError):
    """A RoboMaster action failed or timed out."""


class GraspFailed(RuntimeError):
    """The gripper did not provide evidence that an object was held."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    center_norm: tuple[float, float]


@dataclass(frozen=True)
class GridObservation:
    grid_id: str
    state: str
    class_name: str | None
    confidence: float
    hit_frames: int
    vote_share: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{label} must be finite")
    return float(value)


def _validate_arm_point(point: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(point, list) or len(point) != 2:
        raise ConfigError(f"{label} must be [x_mm, y_mm]")
    x_mm = _check_number(point[0], f"{label}[0]")
    y_mm = _check_number(point[1], f"{label}[1]")
    # moveto() coordinates are relative to the arm's power-on origin, so valid
    # calibrated coordinates can be negative. This is only a gross typo guard;
    # physical reach and collision safety must be established on the real robot.
    limit_mm = _check_number(
        cfg["safety"]["max_abs_arm_coordinate_mm"],
        "safety.max_abs_arm_coordinate_mm",
    )
    if abs(x_mm) > limit_mm or abs(y_mm) > limit_mm:
        raise ConfigError(f"{label} exceeds safety.max_abs_arm_coordinate_mm")


def _validate_route(route: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(route, list):
        raise ConfigError(f"{label} must be a list")
    limits = cfg["safety"]
    for index, step in enumerate(route):
        if not isinstance(step, dict):
            raise ConfigError(f"{label}[{index}] must be an object")
        x_m = _check_number(step.get("x_m", 0), f"{label}[{index}].x_m")
        y_m = _check_number(step.get("y_m", 0), f"{label}[{index}].y_m")
        z_deg = _check_number(step.get("z_deg", 0), f"{label}[{index}].z_deg")
        speed = _check_number(
            step.get("xy_speed_mps", limits["default_xy_speed_mps"]),
            f"{label}[{index}].xy_speed_mps",
        )
        z_speed = _check_number(
            step.get("z_speed_dps", limits["default_z_speed_dps"]),
            f"{label}[{index}].z_speed_dps",
        )
        if abs(x_m) > limits["max_step_m"] or abs(y_m) > limits["max_step_m"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_step_m")
        if abs(z_deg) > limits["max_rotation_deg"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_rotation_deg")
        if not 0.5 <= speed <= 2.0:
            raise ConfigError(f"{label}[{index}] xy speed must be in [0.5, 2.0] m/s")
        if not 10.0 <= z_speed <= 540.0:
            raise ConfigError(f"{label}[{index}] rotation speed must be in [10, 540] deg/s")
        if (x_m or y_m or z_deg) and not limits["chassis_motion_enabled"]:
            raise ConfigError(
                f"{label}[{index}] requests chassis motion while chassis_motion_enabled is false"
            )


def validate_config(cfg: dict[str, Any], require_calibrated: bool = False) -> None:
    required_sections = {
        "robot",
        "camera",
        "model",
        "task",
        "safety",
        "layout",
        "grids",
        "arm_profiles",
        "bins",
    }
    missing = required_sections - set(cfg)
    if missing:
        raise ConfigError("Missing sections: " + ", ".join(sorted(missing)))

    grids = cfg["grids"]
    if not isinstance(grids, list) or len(grids) != 6:
        raise ConfigError("grids must contain exactly six fixed pickup grids")
    grid_ids = [grid.get("id") for grid in grids]
    if any(not isinstance(grid_id, str) or not grid_id for grid_id in grid_ids):
        raise ConfigError("Every grid requires a non-empty string id")
    if len(set(grid_ids)) != len(grid_ids):
        raise ConfigError("Grid ids must be unique")
    if grid_ids != ["G1", "G2", "G3", "G4", "G5", "G6"]:
        raise ConfigError("Grid order must be G1, G2, G3, G4, G5, G6")

    priorities = [grid.get("priority") for grid in grids]
    if priorities != [1, 2, 3, 4, 5, 6]:
        raise ConfigError("Grid priorities must be 1 through 6 in G1-G6 order")

    for grid in grids:
        roi = grid.get("roi_norm")
        if not isinstance(roi, list) or len(roi) != 4:
            raise ConfigError(f"{grid['id']}.roi_norm must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (_check_number(v, f"{grid['id']}.roi_norm") for v in roi)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ConfigError(f"{grid['id']}.roi_norm must be ordered inside [0, 1]")
        if grid.get("row") not in {"front", "rear"}:
            raise ConfigError(f"{grid['id']}.row must be front or rear")
        center = grid.get("ground_center_m")
        if not isinstance(center, dict):
            raise ConfigError(f"{grid['id']}.ground_center_m must be an object")
        _check_number(center.get("x_forward_m"), f"{grid['id']}.ground_center_m.x_forward_m")
        _check_number(center.get("y_right_m"), f"{grid['id']}.ground_center_m.y_right_m")

        blocker = grid.get("blocked_by")
        if blocker is not None and blocker not in grid_ids:
            raise ConfigError(f"{grid['id']}.blocked_by references an unknown grid")
        if blocker == grid["id"]:
            raise ConfigError(f"{grid['id']} cannot block itself")

    for left_index, left in enumerate(grids):
        lx1, ly1, lx2, ly2 = left["roi_norm"]
        for right in grids[left_index + 1 :]:
            rx1, ry1, rx2, ry2 = right["roi_norm"]
            overlaps = min(lx2, rx2) > max(lx1, rx1) and min(ly2, ry2) > max(ly1, ry1)
            if overlaps:
                raise ConfigError(f"Grid ROIs overlap: {left['id']} and {right['id']}")

    targets = cfg["task"].get("target_classes")
    if targets != ["cup", "bottle"]:
        raise ConfigError('task.target_classes must be exactly ["cup", "bottle"]')
    expected = {int(key): value for key, value in cfg["model"]["expected_class_ids"].items()}
    if expected != {0: "cup", 1: "mouse", 2: "bottle"}:
        raise ConfigError("model.expected_class_ids must match the supplied three-class model")
    if cfg["robot"]["conn_type"] not in {"ap", "sta", "rndis"}:
        raise ConfigError("robot.conn_type must be ap, sta, or rndis")
    if cfg["robot"].get("proto_type", "udp") not in {"udp", "tcp"}:
        raise ConfigError("robot.proto_type must be udp or tcp")
    if cfg["camera"]["source"] not in {"robomaster", "usb"}:
        raise ConfigError("camera.source must be robomaster or usb")
    if cfg["camera"].get("resolution") not in {"360p", "540p", "720p"}:
        raise ConfigError("camera.resolution must be 360p, 540p, or 720p")

    layout = cfg["layout"]
    if layout.get("coordinate_frame") != "robot_start_x_forward_y_right":
        raise ConfigError("layout.coordinate_frame must be robot_start_x_forward_y_right")
    _check_number(layout.get("grid_square_size_m"), "layout.grid_square_size_m")
    _check_number(layout.get("nominal_arm_reach_m"), "layout.nominal_arm_reach_m")

    scan_frames = int(cfg["model"]["scan_frames"])
    min_hits = int(cfg["model"]["min_hit_frames"])
    if scan_frames < 1 or not 1 <= min_hits <= scan_frames:
        raise ConfigError("model.min_hit_frames must be between 1 and scan_frames")
    vote_share = float(cfg["model"]["min_vote_share"])
    if not 0.5 <= vote_share <= 1.0:
        raise ConfigError("model.min_vote_share must be in [0.5, 1.0]")

    # Chassis routes are fully specified in the shipped ground layout and can
    # therefore be checked even while arm coordinates remain uncalibrated.
    for grid in grids:
        motion = grid.get("motion", {})
        _validate_route(motion.get("chassis_to_pick", []), f"{grid['id']}.chassis_to_pick", cfg)
        _validate_route(
            motion.get("chassis_home_from_pick", []),
            f"{grid['id']}.chassis_home_from_pick",
            cfg,
        )
    for class_name in targets:
        if class_name not in cfg["bins"]:
            raise ConfigError(f"Missing bin configuration for {class_name}")
        motion = cfg["bins"][class_name]
        _validate_route(motion.get("chassis_to_bin", []), f"bins.{class_name}.chassis_to_bin", cfg)
        _validate_route(
            motion.get("chassis_home_from_bin", []),
            f"bins.{class_name}.chassis_home_from_bin",
            cfg,
        )

    if not require_calibrated:
        return
    if cfg.get("calibrated") is not True:
        raise ConfigError('Real motion is blocked until the config contains "calibrated": true')
    for class_name in targets:
        if class_name not in cfg["arm_profiles"]:
            raise ConfigError(f"Missing pickup arm profile for {class_name}")
        motion = cfg["arm_profiles"][class_name]
        for key in ("arm_pregrasp_mm", "arm_grasp_mm", "arm_lift_mm"):
            _validate_arm_point(motion.get(key), f"arm_profiles.{class_name}.{key}", cfg)
    for class_name in targets:
        motion = cfg["bins"][class_name]
        for key in ("arm_predrop_mm", "arm_drop_mm", "arm_retract_mm"):
            _validate_arm_point(motion.get(key), f"bins.{class_name}.{key}", cfg)


def point_to_grid(center_norm: tuple[float, float], grids: list[dict[str, Any]]) -> str | None:
    x, y = center_norm
    matches = []
    for grid in grids:
        x1, y1, x2, y2 = grid["roi_norm"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            matches.append(grid["id"])
    return matches[0] if len(matches) == 1 else None


def summarize_grid(
    grid_id: str,
    frame_detections: Iterable[Detection | None],
    target_classes: set[str],
    min_hits: int,
    min_vote_share: float,
) -> GridObservation:
    detections = [item for item in frame_detections if item is not None]
    if not detections:
        return GridObservation(grid_id, "empty", None, 0.0, 0, 0.0)

    votes = Counter(item.class_name for item in detections)
    class_name, hit_frames = votes.most_common(1)[0]
    vote_share = hit_frames / len(detections)
    confidence = sum(
        item.confidence for item in detections if item.class_name == class_name
    ) / hit_frames

    if hit_frames < min_hits:
        state = "uncertain"
    elif vote_share < min_vote_share:
        state = "ambiguous"
    elif class_name in target_classes:
        state = "candidate"
    else:
        state = "unsupported"
    return GridObservation(grid_id, state, class_name, confidence, hit_frames, vote_share)


def plan_candidates(
    observations: Iterable[GridObservation],
    grids: list[dict[str, Any]],
    failures: Counter[str],
    max_failures_per_grid: int,
) -> tuple[list[GridObservation], list[dict[str, str]]]:
    """Return safe candidates in configured order and blocked rear-grid reasons.

    A rear grid is reachable only after the front grid in the same driving lane
    is observed empty. A mouse, uncertain observation, or failed pickup in the
    front lane therefore prevents the robot from driving through that lane.
    """
    observations_by_id = {item.grid_id: item for item in observations}
    grids_by_id = {grid["id"]: grid for grid in grids}
    eligible: list[GridObservation] = []
    blocked: list[dict[str, str]] = []

    for observation in observations_by_id.values():
        if observation.state != "candidate":
            continue
        if failures[observation.grid_id] >= max_failures_per_grid:
            continue
        grid = grids_by_id[observation.grid_id]
        blocker_id = grid.get("blocked_by")
        if blocker_id is not None:
            blocker = observations_by_id[blocker_id]
            if blocker.state != "empty":
                blocked.append(
                    {
                        "grid_id": observation.grid_id,
                        "blocked_by": blocker_id,
                        "blocker_state": blocker.state,
                    }
                )
                continue
        eligible.append(observation)

    eligible.sort(key=lambda item: (int(grids_by_id[item.grid_id]["priority"]), -item.confidence))
    return eligible, blocked


class EventLog:
    def __init__(self, directory: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = directory / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "task_log.jsonl"
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, event: str, **details: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **details}
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
        print(json.dumps(record, ensure_ascii=False))

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class Detector:
    def __init__(self, cfg: dict[str, Any], config_path: Path) -> None:
        config_root = ROOT / ".ultralytics"
        try:
            (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
            if not os.access(config_root / "Ultralytics", os.W_OK):
                raise OSError("not writable")
        except OSError:
            config_root = Path(tempfile.gettempdir()) / "robomaster_sorter_ultralytics"
            (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))

        from ultralytics import YOLO

        weights = Path(cfg["model"]["weights"])
        if not weights.is_absolute():
            weights = (config_path.parent / weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.cfg = cfg["model"]

        names = self.model.names
        actual = {int(key): str(value) for key, value in names.items()}
        expected = {int(key): str(value) for key, value in self.cfg["expected_class_ids"].items()}
        if actual != expected:
            raise RuntimeError(f"Model class mapping mismatch: expected {expected}, got {actual}")

    def predict(self, frame: Any) -> list[Detection]:
        height, width = frame.shape[:2]
        result = self.model.predict(
            frame,
            conf=float(self.cfg["confidence"]),
            iou=float(self.cfg["iou"]),
            imgsz=int(self.cfg["imgsz"]),
            device=str(self.cfg["device"]),
            verbose=False,
        )[0]
        output: list[Detection] = []
        if result.boxes is None:
            return output
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for box, class_id_raw, score in zip(boxes, classes, scores):
            class_id = int(class_id_raw)
            cx = (float(box[0]) + float(box[2])) / (2.0 * width)
            cy = (float(box[1]) + float(box[3])) / (2.0 * height)
            output.append(
                Detection(
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    confidence=float(score),
                    xyxy=tuple(float(value) for value in box),
                    center_norm=(cx, cy),
                )
            )
        return output


class RoboMasterController:
    def __init__(self, cfg: dict[str, Any], needs_camera: bool, motion_enabled: bool) -> None:
        import robomaster
        from robomaster import camera as rm_camera
        from robomaster import robot

        local_ip = cfg["robot"].get("local_ip")
        if local_ip:
            robomaster.config.LOCAL_IP_STR = str(local_ip)

        self.cfg = cfg
        self.motion_enabled = motion_enabled
        self.robot = robot.Robot()
        initialize_kwargs: dict[str, Any] = {
            "conn_type": cfg["robot"]["conn_type"],
            "proto_type": cfg["robot"].get("proto_type", "udp"),
        }
        if cfg["robot"].get("sn"):
            initialize_kwargs["sn"] = cfg["robot"]["sn"]
        self._camera_started = False
        self._gripper_status_subscribed = False
        self.gripper_status: str | None = None
        try:
            self.robot.initialize(**initialize_kwargs)
            self.chassis = self.robot.chassis
            self.arm = self.robot.robotic_arm
            self.gripper = self.robot.gripper
            self.camera = self.robot.camera if needs_camera else None

            if needs_camera:
                resolution = {
                    "360p": rm_camera.STREAM_360P,
                    "540p": rm_camera.STREAM_540P,
                    "720p": rm_camera.STREAM_720P,
                }[cfg["camera"]["resolution"]]
                started = self.camera.start_video_stream(display=False, resolution=resolution)
                if started is False:
                    raise RuntimeError("RoboMaster video stream failed to start")
                self._camera_started = True
        except Exception:
            self.robot.close()
            raise

    def enable_gripper_status(self) -> None:
        def update(status: str) -> None:
            self.gripper_status = status

        if not self.gripper.sub_status(freq=5, callback=update):
            raise RuntimeError("Cannot subscribe to gripper status")
        self._gripper_status_subscribed = True
        time.sleep(0.5)

    def read_frame(self) -> Any:
        assert self.camera is not None
        frame = self.camera.read_cv2_image(timeout=3, strategy="newest")
        if frame is None:
            raise RuntimeError("Timed out while reading the RoboMaster camera")
        return frame

    def wait_action(self, action: Any, label: str) -> None:
        timeout = float(self.cfg["safety"]["action_timeout_s"])
        completed = action.wait_for_completed(timeout=timeout)
        if not completed or not bool(getattr(action, "has_succeeded", False)):
            reason = getattr(action, "failure_reason", None)
            raise MotionError(f"{label} failed or timed out; reason={reason!r}")

    def move_arm(self, point: list[float], label: str) -> None:
        self.wait_action(self.arm.moveto(x=point[0], y=point[1]), label)

    def recenter_arm(self) -> None:
        self.wait_action(self.arm.recenter(), "arm recenter")

    def run_route(self, route: list[dict[str, Any]], label: str) -> None:
        limits = self.cfg["safety"]
        for index, step in enumerate(route):
            x_m = float(step.get("x_m", 0))
            y_m = float(step.get("y_m", 0))
            z_deg = float(step.get("z_deg", 0))
            if x_m == y_m == z_deg == 0:
                continue
            action = self.chassis.move(
                x=x_m,
                y=y_m,
                z=z_deg,
                xy_speed=float(step.get("xy_speed_mps", limits["default_xy_speed_mps"])),
                z_speed=float(step.get("z_speed_dps", limits["default_z_speed_dps"])),
            )
            self.wait_action(action, f"{label} step {index + 1}")

    def open_gripper(self) -> None:
        power = int(self.cfg["safety"]["gripper_open_power"])
        if self.gripper.open(power=power) is False:
            raise MotionError("gripper open command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_open_seconds"]))
        self.gripper.pause()
        time.sleep(0.2)

    def close_gripper(self) -> None:
        power = int(self.cfg["safety"]["gripper_close_power"])
        if self.gripper.close(power=power) is False:
            raise MotionError("gripper close command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_close_seconds"]))
        self.gripper.pause()
        time.sleep(0.4)

    def verify_grasp(self) -> None:
        verify_cfg = self.cfg["safety"]["grasp_verification"]
        if not verify_cfg["enabled"]:
            return
        if self.gripper_status not in set(verify_cfg["success_states"]):
            raise GraspFailed(f"gripper status is {self.gripper_status!r}")

    def stop_safely(self) -> None:
        if not self.motion_enabled:
            return
        try:
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        except Exception:
            pass
        try:
            self.gripper.pause()
        except Exception:
            pass

    def close(self) -> None:
        if self.motion_enabled:
            self.stop_safely()
        if self._gripper_status_subscribed:
            try:
                self.gripper.unsub_status()
            except Exception:
                pass
        if self._camera_started and self.camera is not None:
            try:
                self.camera.stop_video_stream()
            except Exception:
                pass
        self.robot.close()


class UsbCamera:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import cv2

        camera_cfg = cfg["camera"]
        self.capture = cv2.VideoCapture(int(camera_cfg["usb_index"]))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open USB camera index {camera_cfg['usb_index']}")

    def read_frame(self) -> Any:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Cannot read a frame from the USB camera")
        return frame

    def close(self) -> None:
        self.capture.release()


def draw_scene(frame: Any, grids: list[dict[str, Any]], detections: list[Detection], observations: list[GridObservation]) -> Any:
    import cv2

    image = frame.copy()
    height, width = image.shape[:2]
    states = {item.grid_id: item for item in observations}
    colors = {
        "candidate": (0, 200, 0),
        "empty": (180, 180, 180),
        "unsupported": (0, 165, 255),
        "uncertain": (0, 0, 255),
        "ambiguous": (255, 0, 255),
    }
    for grid in grids:
        x1, y1, x2, y2 = grid["roi_norm"]
        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        observation = states.get(grid["id"])
        state = observation.state if observation else "empty"
        label = f"{grid['id']}: {state}"
        if observation and observation.class_name:
            label += f" {observation.class_name} {observation.confidence:.2f}"
        cv2.rectangle(image, p1, p2, colors[state], 2)
        cv2.putText(image, label, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[state], 2)
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            image,
            f"{detection.class_name} {detection.confidence:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
    return image


def scan_scene(
    detector: Detector,
    read_frame: Callable[[], Any],
    cfg: dict[str, Any],
    event_log: EventLog,
    scan_index: int,
    show: bool,
) -> list[GridObservation]:
    import cv2

    grids = cfg["grids"]
    evidence: dict[str, list[Detection | None]] = defaultdict(list)
    latest_frame = None
    latest_detections: list[Detection] = []

    for _ in range(int(cfg["camera"].get("warmup_frames", 2))):
        read_frame()

    for frame_index in range(int(cfg["model"]["scan_frames"])):
        frame = read_frame()
        detections = detector.predict(frame)
        by_grid: dict[str, list[Detection]] = defaultdict(list)
        for detection in detections:
            grid_id = point_to_grid(detection.center_norm, grids)
            if grid_id is not None:
                by_grid[grid_id].append(detection)
            else:
                event_log.write("detection_outside_grid", scan=scan_index, detection=asdict(detection))
        for grid in grids:
            choices = by_grid.get(grid["id"], [])
            if len(choices) > 1:
                event_log.write(
                    "multiple_detections_in_grid",
                    scan=scan_index,
                    frame=frame_index,
                    grid_id=grid["id"],
                    detections=[asdict(item) for item in choices],
                )
            evidence[grid["id"]].append(max(choices, key=lambda item: item.confidence) if choices else None)
        latest_frame = frame
        latest_detections = detections

    targets = set(cfg["task"]["target_classes"])
    observations = [
        summarize_grid(
            grid["id"],
            evidence[grid["id"]],
            targets,
            int(cfg["model"]["min_hit_frames"]),
            float(cfg["model"]["min_vote_share"]),
        )
        for grid in grids
    ]
    event_log.write("scan_complete", scan=scan_index, observations=[asdict(item) for item in observations])

    assert latest_frame is not None
    annotated = draw_scene(latest_frame, grids, latest_detections, observations)
    image_path = event_log.run_dir / f"scan_{scan_index:03d}.jpg"
    if not cv2.imwrite(str(image_path), annotated):
        raise RuntimeError(f"Cannot save scan evidence image: {image_path}")
    if show:
        cv2.imshow("RoboMaster cup/bottle sorter", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
    return observations


def sort_one(
    controller: RoboMasterController,
    grid: dict[str, Any],
    class_name: str,
    cfg: dict[str, Any],
    event_log: EventLog,
) -> None:
    route = grid["motion"]
    pick = cfg["arm_profiles"][class_name]
    drop = cfg["bins"][class_name]
    event_log.write(
        "sort_started",
        grid_id=grid["id"],
        class_name=class_name,
        priority=grid["priority"],
        ground_center_m=grid["ground_center_m"],
    )

    controller.recenter_arm()
    controller.run_route(route.get("chassis_to_pick", []), f"route to {grid['id']}")
    controller.open_gripper()
    controller.move_arm(pick["arm_pregrasp_mm"], f"{grid['id']} pregrasp")
    controller.move_arm(pick["arm_grasp_mm"], f"{grid['id']} grasp")
    controller.close_gripper()
    controller.verify_grasp()
    controller.move_arm(pick["arm_lift_mm"], f"{grid['id']} lift")
    controller.recenter_arm()
    controller.run_route(route.get("chassis_home_from_pick", []), f"return from {grid['id']}")

    controller.run_route(drop.get("chassis_to_bin", []), f"route to {class_name} bin")
    controller.move_arm(drop["arm_predrop_mm"], f"{class_name} predrop")
    controller.move_arm(drop["arm_drop_mm"], f"{class_name} drop")
    controller.open_gripper()
    controller.move_arm(drop["arm_retract_mm"], f"{class_name} retract")
    controller.recenter_arm()
    controller.run_route(drop.get("chassis_home_from_bin", []), f"return from {class_name} bin")
    event_log.write("sort_completed", grid_id=grid["id"], class_name=class_name)


def recover_empty_grasp(
    controller: RoboMasterController,
    grid: dict[str, Any],
    event_log: EventLog,
) -> None:
    """Recover only from a verified empty grasp while the commanded pose is still known."""
    try:
        controller.open_gripper()
        controller.recenter_arm()
        controller.run_route(grid["motion"].get("chassis_home_from_pick", []), f"recover from {grid['id']}")
        event_log.write("grasp_recovery_completed", grid_id=grid["id"])
    except Exception as error:
        controller.stop_safely()
        raise MotionError(f"Failed to recover from empty grasp: {error}") from error


def run_task(
    detector: Detector,
    read_frame: Callable[[], Any],
    controller: RoboMasterController | None,
    cfg: dict[str, Any],
    event_log: EventLog,
    execute: bool,
    show: bool,
    max_objects: int,
) -> dict[str, Any]:
    grid_by_id = {grid["id"]: grid for grid in cfg["grids"]}
    failures: Counter[str] = Counter()
    sorted_counts: Counter[str] = Counter()
    scan_index = 0
    no_candidate_rounds = 0

    while sum(sorted_counts.values()) < max_objects:
        scan_index += 1
        observations = scan_scene(detector, read_frame, cfg, event_log, scan_index, show)
        candidates, blocked = plan_candidates(
            observations,
            cfg["grids"],
            failures,
            int(cfg["task"]["max_failures_per_grid"]),
        )
        if blocked:
            event_log.write("rear_lanes_blocked", blocked=blocked)
        if not execute:
            event_log.write(
                "dry_run_complete",
                planned=[asdict(item) for item in candidates],
                blocked=blocked,
            )
            break
        if not candidates:
            no_candidate_rounds += 1
            event_log.write("no_executable_target", consecutive_rounds=no_candidate_rounds)
            if no_candidate_rounds >= int(cfg["task"]["stop_after_no_candidate_scans"]):
                break
            continue

        no_candidate_rounds = 0
        target = candidates[0]
        assert controller is not None
        assert target.class_name is not None
        grid = grid_by_id[target.grid_id]
        try:
            sort_one(controller, grid, target.class_name, cfg, event_log)
            # Return to the same camera reference pose and verify the source grid
            # before counting a completed cycle. This does not prove the object is
            # inside the bin, but it prevents counting a visibly failed pickup.
            scan_index += 1
            verification = scan_scene(
                detector, read_frame, cfg, event_log, scan_index, show
            )
            verified_grid = next(item for item in verification if item.grid_id == target.grid_id)
            if verified_grid.state == "empty":
                sorted_counts[target.class_name] += 1
                event_log.write(
                    "source_grid_cleared",
                    grid_id=target.grid_id,
                    class_name=target.class_name,
                )
            else:
                failures[target.grid_id] += 1
                event_log.write(
                    "post_action_verification_failed",
                    grid_id=target.grid_id,
                    expected="empty",
                    observed=asdict(verified_grid),
                )
        except GraspFailed as error:
            failures[target.grid_id] += 1
            event_log.write("grasp_failed", grid_id=target.grid_id, error=str(error))
            recover_empty_grasp(controller, grid, event_log)
        except Exception as error:
            controller.stop_safely()
            event_log.write("fatal_motion_error", grid_id=target.grid_id, error=repr(error))
            raise

    summary = {
        "finished_at": utc_now(),
        "execute": execute,
        "scans": scan_index,
        "sorted_total": sum(sorted_counts.values()),
        "sorted_by_class": dict(sorted_counts),
        "failures_by_grid": dict(failures),
        "stop_reason": "max_objects" if sum(sorted_counts.values()) >= max_objects else "no_executable_target_or_dry_run",
    }
    (event_log.run_dir / "task_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    event_log.write("task_finished", **summary)
    return summary


def run_chassis_route_test(cfg: dict[str, Any], target: str) -> None:
    """Execute exactly one outbound/return chassis route without arm motion."""
    if target in {"cup-bin", "bottle-bin"}:
        class_name = target[:-4] if target.endswith("-bin") else target
        route_out = cfg["bins"][class_name]["chassis_to_bin"]
        route_home = cfg["bins"][class_name]["chassis_home_from_bin"]
    else:
        grid = next(grid for grid in cfg["grids"] if grid["id"] == target)
        route_out = grid["motion"]["chassis_to_pick"]
        route_home = grid["motion"]["chassis_home_from_pick"]

    print(f"Testing chassis route {target}; keep the floor path clear and prepare physical stop.")
    controller = RoboMasterController(cfg, needs_camera=False, motion_enabled=True)
    try:
        controller.run_route(route_out, f"test outbound {target}")
        time.sleep(1.0)
        controller.run_route(route_home, f"test return {target}")
        print(f"Chassis route {target} completed and commanded return to the start pose.")
    finally:
        controller.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Enable calibrated robot movement")
    parser.add_argument("--preview-only", action="store_true", help="Alias for one no-motion scan")
    parser.add_argument("--validate-config", action="store_true", help="Validate JSON and exit")
    parser.add_argument(
        "--test-route",
        choices=["G1", "G2", "G3", "G4", "G5", "G6", "cup-bin", "bottle-bin"],
        help="With --execute, test one chassis outbound/return route without arm motion",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)

    if args.test_route is not None:
        if not args.execute or args.preview_only or args.validate_config:
            raise ConfigError("--test-route requires --execute and cannot be combined with preview/validation")
        validate_config(cfg, require_calibrated=False)
        run_chassis_route_test(cfg, args.test_route)
        return

    execute = bool(args.execute and not args.preview_only)
    validate_config(cfg, require_calibrated=execute)
    if args.validate_config:
        print(f"Configuration is valid for {'execution' if execute else 'dry-run'}: {config_path}")
        return

    max_objects = (
        args.max_objects
        if args.max_objects is not None
        else int(cfg["task"]["max_objects"])
    )
    if max_objects < 1:
        raise ConfigError("max_objects must be at least 1")

    event_log = EventLog(args.log_dir.resolve())
    controller: RoboMasterController | None = None
    usb_camera: UsbCamera | None = None
    try:
        event_log.write(
            "task_started",
            mode="execute" if execute else "dry_run",
            config=str(config_path),
            max_objects=max_objects,
        )
        detector = Detector(cfg, config_path)
        if cfg["camera"]["source"] == "robomaster":
            controller = RoboMasterController(cfg, needs_camera=True, motion_enabled=execute)
            read_frame = controller.read_frame
        else:
            usb_camera = UsbCamera(cfg)
            read_frame = usb_camera.read_frame
            if execute:
                controller = RoboMasterController(cfg, needs_camera=False, motion_enabled=True)

        if execute:
            assert controller is not None
            controller.enable_gripper_status()
        summary = run_task(
            detector,
            read_frame,
            controller,
            cfg,
            event_log,
            execute,
            not args.no_display,
            max_objects,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        event_log.write("operator_stop")
        if controller is not None:
            controller.stop_safely()
    except Exception as error:
        event_log.write("task_exception", error=repr(error))
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
