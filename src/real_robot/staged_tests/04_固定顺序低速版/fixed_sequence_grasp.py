"""Run six fixed RoboMaster grasps without camera or object recognition."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "fixed_sequence_config.json"
ROBOT_AP_IP = "192.168.2.1"
ROBOT_AP_NETWORK = ipaddress.ip_network("192.168.2.0/24")
EXPECTED_SEQUENCE = [
    ("G2", "left"),
    ("G5", "right"),
    ("G1", "right"),
    ("G4", "right"),
    ("G3", "right"),
    ("G6", "right"),
]


class ConfigError(ValueError):
    """The fixed-sequence configuration is incomplete or unsafe."""


class MotionError(RuntimeError):
    """A RoboMaster command failed or timed out."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{label} must be finite")
    return number


def validate_arm_point(point: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(point, list) or len(point) != 2:
        raise ConfigError(f"{label} must be [x_mm, y_mm]")
    limit = check_number(
        cfg["safety"]["max_abs_arm_coordinate_mm"],
        "safety.max_abs_arm_coordinate_mm",
    )
    for index, value in enumerate(point):
        coordinate = check_number(value, f"{label}[{index}]")
        if abs(coordinate) > limit:
            raise ConfigError(f"{label} exceeds safety.max_abs_arm_coordinate_mm")


def validate_route(route: Any, label: str, cfg: dict[str, Any]) -> None:
    if not isinstance(route, list):
        raise ConfigError(f"{label} must be a list")
    safety = cfg["safety"]
    default_speed = check_number(
        safety["translation_speed_mps"], "safety.translation_speed_mps"
    )
    for index, step in enumerate(route):
        if not isinstance(step, dict):
            raise ConfigError(f"{label}[{index}] must be an object")
        x_m = check_number(step.get("x_m", 0), f"{label}[{index}].x_m")
        y_m = check_number(step.get("y_m", 0), f"{label}[{index}].y_m")
        z_deg = check_number(step.get("z_deg", 0), f"{label}[{index}].z_deg")
        nonzero_axes = sum(value != 0 for value in (x_m, y_m, z_deg))
        if nonzero_axes != 1:
            raise ConfigError(f"{label}[{index}] must move exactly one axis")
        if abs(x_m) > safety["max_step_m"] or abs(y_m) > safety["max_step_m"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_step_m")
        if abs(z_deg) > safety["max_rotation_deg"]:
            raise ConfigError(f"{label}[{index}] exceeds safety.max_rotation_deg")
        speed = check_number(
            step.get("translation_speed_mps", default_speed),
            f"{label}[{index}].translation_speed_mps",
        )
        if (x_m or y_m) and not 0.10 <= speed < 0.50:
            raise ConfigError(
                f"{label}[{index}] low-speed translation must be in [0.10, 0.50) m/s"
            )
        z_speed = check_number(
            step.get("z_speed_dps", safety["default_z_speed_dps"]),
            f"{label}[{index}].z_speed_dps",
        )
        if z_deg and not 10.0 <= z_speed <= 540.0:
            raise ConfigError(f"{label}[{index}] z speed must be in [10, 540] deg/s")
        if safety["chassis_motion_enabled"] is not True:
            raise ConfigError(f"{label}[{index}] requests disabled chassis motion")


def validate_config(cfg: dict[str, Any], require_calibrated: bool = False) -> None:
    required = {
        "robot",
        "safety",
        "task",
        "pickup_profiles",
        "grids",
        "destinations",
    }
    missing = required - set(cfg)
    if missing:
        raise ConfigError("Missing sections: " + ", ".join(sorted(missing)))
    if cfg["robot"].get("conn_type") not in {"ap", "sta", "rndis"}:
        raise ConfigError("robot.conn_type must be ap, sta, or rndis")
    if cfg["robot"].get("proto_type", "udp") not in {"udp", "tcp"}:
        raise ConfigError("robot.proto_type must be udp or tcp")

    safety = cfg["safety"]
    translation_speed = check_number(
        safety.get("translation_speed_mps"), "safety.translation_speed_mps"
    )
    if not 0.10 <= translation_speed < 0.50:
        raise ConfigError("translation_speed_mps must be in [0.10, 0.50) m/s")
    z_speed = check_number(
        safety.get("default_z_speed_dps"), "safety.default_z_speed_dps"
    )
    if not 10.0 <= z_speed <= 540.0:
        raise ConfigError("default_z_speed_dps must be in [10, 540] deg/s")
    for key in (
        "startup_delay_s",
        "chassis_step_settle_s",
        "camera_home_settle_s",
        "gripper_open_seconds",
        "gripper_close_seconds",
        "post_close_pause_s",
    ):
        if check_number(safety.get(key), f"safety.{key}") < 0:
            raise ConfigError(f"safety.{key} cannot be negative")

    sequence = cfg["task"].get("fixed_sequence")
    actual = (
        [(step.get("grid_id"), step.get("destination")) for step in sequence]
        if isinstance(sequence, list)
        else []
    )
    if actual != EXPECTED_SEQUENCE:
        raise ConfigError("fixed_sequence must be G2-G5-G1-G4-G3-G6; only G2 goes left")

    grids = cfg["grids"]
    if not isinstance(grids, list) or {grid.get("id") for grid in grids} != {
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
    }:
        raise ConfigError("grids must contain G1 through G6 exactly once")
    for grid in grids:
        center = grid.get("ground_center_m")
        if not isinstance(center, dict):
            raise ConfigError(f"{grid['id']}.ground_center_m must be an object")
        check_number(center.get("x_forward_m"), f"{grid['id']}.x_forward_m")
        check_number(center.get("y_right_m"), f"{grid['id']}.y_right_m")
        validate_route(grid.get("chassis_to_pick"), f"{grid['id']}.chassis_to_pick", cfg)
        validate_route(
            grid.get("chassis_home_from_pick"),
            f"{grid['id']}.chassis_home_from_pick",
            cfg,
        )

    profiles = cfg["pickup_profiles"]
    destinations = cfg["destinations"]
    if set(profiles) != {"left", "right"}:
        raise ConfigError("pickup_profiles must contain exactly left and right")
    if set(destinations) != {"left", "right"}:
        raise ConfigError("destinations must contain exactly left and right")
    for name, profile in profiles.items():
        for key in ("arm_pregrasp_mm", "arm_grasp_mm", "arm_lift_mm"):
            validate_arm_point(profile.get(key), f"pickup_profiles.{name}.{key}", cfg)
    for name, destination in destinations.items():
        validate_route(
            destination.get("chassis_to_destination"),
            f"destinations.{name}.chassis_to_destination",
            cfg,
        )
        validate_route(
            destination.get("chassis_home_from_destination"),
            f"destinations.{name}.chassis_home_from_destination",
            cfg,
        )
        for key in ("arm_predrop_mm", "arm_drop_mm", "arm_retract_mm"):
            validate_arm_point(destination.get(key), f"destinations.{name}.{key}", cfg)

    if require_calibrated and cfg.get("calibrated") is not True:
        raise ConfigError(
            'Real motion is blocked until fixed_sequence_config.json contains "calibrated": true'
        )


def resolve_sdk_connection_types(
    robot_cfg: dict[str, Any], conn_module: Any
) -> tuple[str, str]:
    conn_types = {
        "ap": conn_module.CONNECTION_WIFI_AP,
        "sta": conn_module.CONNECTION_WIFI_STA,
        "rndis": conn_module.CONNECTION_USB_RNDIS,
    }
    proto_types = {
        "udp": conn_module.CONNECTION_PROTO_UDP,
        "tcp": conn_module.CONNECTION_PROTO_TCP,
    }
    return (
        conn_types[robot_cfg["conn_type"]],
        proto_types[robot_cfg.get("proto_type", "udp")],
    )


def validate_ap_local_ip(local_ip: str) -> str:
    try:
        address = ipaddress.ip_address(local_ip)
    except ValueError as error:
        raise MotionError(f"Invalid local IP address: {local_ip!r}") from error
    if (
        address not in ROBOT_AP_NETWORK
        or address == ROBOT_AP_NETWORK.network_address
        or address == ROBOT_AP_NETWORK.broadcast_address
        or str(address) == ROBOT_AP_IP
    ):
        raise MotionError(
            "AP mode requires this computer to use a 192.168.2.x address "
            f"other than {ROBOT_AP_IP}; detected {local_ip}."
        )
    return str(address)


def detect_ap_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((ROBOT_AP_IP, 9))
            local_ip = probe.getsockname()[0]
    except OSError as error:
        raise MotionError(
            "No IPv4 route to the RoboMaster AP. Connect to the RoboMaster Wi-Fi first."
        ) from error
    return validate_ap_local_ip(local_ip)


class EventLog:
    def __init__(self, root: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "fixed_sequence_log.jsonl"
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, event: str, **details: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **details}
        line = json.dumps(record, ensure_ascii=False)
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line)

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class RoboMasterController:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import robomaster
        from robomaster import conn, robot

        local_ip = cfg["robot"].get("local_ip")
        if cfg["robot"]["conn_type"] == "ap":
            local_ip = (
                validate_ap_local_ip(str(local_ip)) if local_ip else detect_ap_local_ip()
            )
        if local_ip:
            robomaster.config.LOCAL_IP_STR = str(local_ip)

        self.cfg = cfg
        self.robot = robot.Robot()
        self.chassis = None
        self.arm = None
        self.gripper = None
        conn_type, proto_type = resolve_sdk_connection_types(cfg["robot"], conn)
        kwargs: dict[str, Any] = {"conn_type": conn_type, "proto_type": proto_type}
        if cfg["robot"].get("sn"):
            kwargs["sn"] = cfg["robot"]["sn"]
        try:
            initialized = self.robot.initialize(**kwargs)
            if initialized is False:
                raise MotionError("RoboMaster initialization returned False")
            self.chassis = self.robot.chassis
            self.arm = self.robot.robotic_arm
            self.gripper = self.robot.gripper
        except Exception as error:
            try:
                self.robot.close()
            except Exception:
                pass
            raise MotionError(
                "RoboMaster SDK initialization failed. Check power, network, and connection mode."
            ) from error

    def wait_action(self, action: Any, label: str) -> None:
        timeout = float(self.cfg["safety"]["action_timeout_s"])
        completed = action.wait_for_completed(timeout=timeout)
        if not completed or not bool(getattr(action, "has_succeeded", False)):
            reason = getattr(action, "failure_reason", None)
            raise MotionError(f"{label} failed or timed out; reason={reason!r}")

    def drive_translation(self, x_m: float, y_m: float, speed: float, label: str) -> None:
        """Translate below chassis.move's 0.5 m/s floor using a timed speed command."""
        assert self.chassis is not None
        distance = math.hypot(x_m, y_m)
        if distance == 0:
            return
        duration = distance / speed
        timeout = max(1, math.ceil(duration + 1.0))
        x_speed = speed * x_m / distance
        y_speed = speed * y_m / distance
        started = self.chassis.drive_speed(
            x=x_speed,
            y=y_speed,
            z=0,
            timeout=timeout,
        )
        if started is False:
            raise MotionError(f"{label} low-speed translation was rejected")
        try:
            time.sleep(duration)
        finally:
            stopped = self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            if stopped is False:
                raise MotionError(f"{label} stop command was rejected")

    def run_route(self, route: list[dict[str, Any]], label: str) -> None:
        assert self.chassis is not None
        safety = self.cfg["safety"]
        for index, step in enumerate(route):
            x_m = float(step.get("x_m", 0))
            y_m = float(step.get("y_m", 0))
            z_deg = float(step.get("z_deg", 0))
            step_label = f"{label} step {index + 1}"
            if x_m or y_m:
                speed = float(
                    step.get("translation_speed_mps", safety["translation_speed_mps"])
                )
                self.drive_translation(x_m, y_m, speed, step_label)
            else:
                action = self.chassis.move(
                    x=0,
                    y=0,
                    z=z_deg,
                    xy_speed=0.5,
                    z_speed=float(
                        step.get("z_speed_dps", safety["default_z_speed_dps"])
                    ),
                )
                self.wait_action(action, step_label)
            settle_s = float(safety["chassis_step_settle_s"])
            if settle_s > 0:
                time.sleep(settle_s)

    def recenter_arm(self, label: str = "arm recenter") -> None:
        assert self.arm is not None
        self.wait_action(self.arm.recenter(), label)

    def restore_camera_home(self, label: str) -> None:
        self.recenter_arm(label)
        settle_s = float(self.cfg["safety"]["camera_home_settle_s"])
        if settle_s > 0:
            time.sleep(settle_s)

    def move_arm(self, point: list[float], label: str) -> None:
        assert self.arm is not None
        self.wait_action(self.arm.moveto(x=point[0], y=point[1]), label)

    def open_gripper(self) -> None:
        assert self.gripper is not None
        power = int(self.cfg["safety"]["gripper_open_power"])
        if self.gripper.open(power=power) is False:
            raise MotionError("gripper open command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_open_seconds"]))
        if self.gripper.pause() is False:
            raise MotionError("gripper pause after open was rejected")
        time.sleep(0.2)

    def close_gripper(self) -> None:
        assert self.gripper is not None
        power = int(self.cfg["safety"]["gripper_close_power"])
        if self.gripper.close(power=power) is False:
            raise MotionError("gripper close command was rejected")
        time.sleep(float(self.cfg["safety"]["gripper_close_seconds"]))
        if self.gripper.pause() is False:
            raise MotionError("gripper pause after close was rejected")
        time.sleep(float(self.cfg["safety"]["post_close_pause_s"]))

    def stop_safely(self) -> None:
        if self.chassis is not None:
            try:
                self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            except Exception:
                pass
        if self.gripper is not None:
            try:
                self.gripper.pause()
            except Exception:
                pass

    def close(self) -> None:
        self.stop_safely()
        try:
            self.robot.close()
        except Exception:
            pass


def execute_position(
    controller: RoboMasterController,
    grid: dict[str, Any],
    destination_name: str,
    cfg: dict[str, Any],
    event_log: EventLog,
) -> None:
    pickup = cfg["pickup_profiles"][destination_name]
    destination = cfg["destinations"][destination_name]
    event_log.write(
        "position_started",
        grid_id=grid["id"],
        destination=destination_name,
        ground_center_m=grid["ground_center_m"],
    )

    controller.restore_camera_home(f"{grid['id']} camera home before pickup")
    controller.run_route(grid["chassis_to_pick"], f"route to {grid['id']}")
    controller.open_gripper()
    controller.move_arm(pickup["arm_pregrasp_mm"], f"{grid['id']} pregrasp")
    controller.move_arm(pickup["arm_grasp_mm"], f"{grid['id']} grasp")
    controller.close_gripper()
    controller.move_arm(pickup["arm_lift_mm"], f"{grid['id']} lift")
    controller.recenter_arm(f"{grid['id']} arm recenter after pickup")
    controller.run_route(grid["chassis_home_from_pick"], f"return from {grid['id']}")

    controller.run_route(
        destination["chassis_to_destination"], f"route to {destination_name}"
    )
    controller.move_arm(destination["arm_predrop_mm"], f"{destination_name} predrop")
    controller.move_arm(destination["arm_drop_mm"], f"{destination_name} drop")
    controller.open_gripper()
    controller.move_arm(destination["arm_retract_mm"], f"{destination_name} retract")
    controller.recenter_arm(f"{destination_name} arm recenter after drop")
    controller.run_route(
        destination["chassis_home_from_destination"],
        f"return from {destination_name}",
    )
    controller.restore_camera_home(f"{grid['id']} camera home after task")
    event_log.write(
        "position_completed", grid_id=grid["id"], destination=destination_name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Run real robot motion")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    validate_config(cfg, require_calibrated=args.execute)
    sequence = cfg["task"]["fixed_sequence"]

    if args.validate_config:
        print(f"Configuration is valid: {config_path}")
        return
    if not args.execute:
        print("No robot motion. Planned fixed sequence:")
        for index, step in enumerate(sequence, start=1):
            print(f"{index}. {step['grid_id']} -> {step['destination']}")
        print(
            f"Translation speed: {cfg['safety']['translation_speed_mps']:.2f} m/s; "
            "camera/video and recognition are disabled."
        )
        return

    event_log = EventLog(args.log_dir.resolve())
    controller: RoboMasterController | None = None
    try:
        controller = RoboMasterController(cfg)
        startup_delay = float(cfg["safety"]["startup_delay_s"])
        print(f"Fixed six-grasp motion starts in {startup_delay:.1f} seconds...")
        if startup_delay > 0:
            time.sleep(startup_delay)
        grids = {grid["id"]: grid for grid in cfg["grids"]}
        event_log.write(
            "fixed_sequence_started",
            fixed_sequence=sequence,
            translation_speed_mps=cfg["safety"]["translation_speed_mps"],
        )
        for index, step in enumerate(sequence, start=1):
            event_log.write("sequence_step", index=index, total=len(sequence))
            execute_position(
                controller,
                grids[step["grid_id"]],
                step["destination"],
                cfg,
                event_log,
            )
        event_log.write("fixed_sequence_completed", positions_completed=len(sequence))
    except KeyboardInterrupt:
        event_log.write("operator_stop")
    except Exception as error:
        event_log.write("fixed_sequence_exception", error=repr(error))
        raise
    finally:
        if controller is not None:
            controller.close()
        event_log.close()


if __name__ == "__main__":
    main()
